from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.datautils import add_camels_attributes, load_attributes

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMELS_ROOT = ROOT / "data" / "camels_root"
DEFAULT_EVALUATION_ROOT = ROOT / "outputs" / "evaluation"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "spatial_shap_group_count"
DEFAULT_ST_ROOT = ROOT / "outputs" / "cluster_lstm"
DEFAULT_PTST_ROOT = ROOT / "outputs" / "pretrain_finetune_lstm"
DEFAULT_MST_ROOT = ROOT / "outputs" / "meta_finetune_lstm"
DEFAULT_SEEDS = (12345, 1234, 123, 12, 1)
GROUP_COUNT_COLUMN = "group_training_basin_count"

MODEL_ORDER = ["global", "cluster", "pretrain_finetune", "meta_finetune"]
MODEL_LABELS = {
    "global": "G",
    "cluster": "ST",
    "pretrain_finetune": "PTST",
    "meta_finetune": "MST",
}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 12345,
    "n_jobs": -1,
}


def normalize_basin_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if not digits:
        raise ValueError(f"Invalid basin id: {value!r}")
    return digits.zfill(8)


def load_metrics(model_name: str, evaluation_root: Path, source: str) -> tuple[pd.DataFrame, str]:
    model_dir = evaluation_root / model_name
    ensemble_file = model_dir / "basin_metrics_ensemble.csv"
    all_seed_file = model_dir / "basin_metrics_all_seeds.csv"

    if source == "ensemble":
        if not ensemble_file.exists():
            raise FileNotFoundError(
                f"Requested the daily-prediction ensemble, but {ensemble_file} does not exist. "
                "Run evaluate_models.py for a genuine ensemble, or explicitly use "
                "--performance_source median (not equivalent to daily ensemble)."
            )
        df = pd.read_csv(ensemble_file, dtype={"basin_id": str})
        return df, "ensemble"

    if source == "median":
        if not all_seed_file.exists():
            raise FileNotFoundError(all_seed_file)
        df = pd.read_csv(all_seed_file, dtype={"basin_id": str})
        return aggregate_seed_metrics(df), "median_across_seeds"

    if source.startswith("seed"):
        seed = int(source.replace("seed", ""))
        seed_file = model_dir / f"basin_metrics_seed{seed}.csv"
        if not seed_file.exists():
            raise FileNotFoundError(seed_file)
        df = pd.read_csv(seed_file, dtype={"basin_id": str})
        return df, f"seed{seed}"

    raise ValueError(source)


def aggregate_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["basin_id"] = df["basin_id"].map(normalize_basin_id)
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"seed", "cluster_id"}
    ]
    agg = df.groupby("basin_id", as_index=False)[numeric_cols].median()
    fold_map = df.groupby("basin_id")["fold"].first().reset_index() if "fold" in df.columns else None
    if fold_map is not None and "fold" not in agg.columns:
        agg = agg.merge(fold_map, on="basin_id", how="left")
    return agg


def load_group_training_counts(
    model_name: str,
    evaluation_root: Path,
    metric_source: str,
    expected_basins: list[str],
    st_root: Path,
    ptst_root: Path,
    mst_root: Path,
    ensemble_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Obtain actual specialist/finetuning basin counts for each evaluated target.

    The basin->expert mapping is read from evaluate_models.py's *per-seed*
    evaluation rows, not inferred from target-basin discharge or an arbitrary
    reference seed. For ensemble results, the feature is the arithmetic mean
    of the five experts actually used. For a metric median across seeds, it
    is the median of the corresponding group sizes. For a single seed it is
    the group size of that one expert.
    """
    if model_name not in {"cluster", "pretrain_finetune", "meta_finetune"}:
        raise ValueError(f"Group training count does not apply to {model_name}")

    roots = {
        "cluster": st_root,
        "pretrain_finetune": ptst_root,
        "meta_finetune": mst_root,
    }
    model_root = roots[model_name]
    metrics_root = evaluation_root / model_name
    if metric_source.startswith("seed"):
        selected_seeds = [int(metric_source[4:])]
        path = metrics_root / f"basin_metrics_seed{selected_seeds[0]}.csv"
    else:
        selected_seeds = [int(x) for x in ensemble_seeds]
        path = metrics_root / "basin_metrics_all_seeds.csv"
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("--ensemble_seeds must list distinct seeds")
    if not path.exists():
        raise FileNotFoundError(f"Training-group SHAP feature needs per-seed evaluation records: {path}")

    df = pd.read_csv(path, dtype={"basin_id": str})
    required_cols = {"basin_id", "seed", "fold", "cluster_id"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"{path}: missing required columns {sorted(required_cols - set(df.columns))}")
    df = df[["basin_id", "seed", "fold", "cluster_id"]].copy()
    df["basin_id"] = df["basin_id"].map(normalize_basin_id)
    for col in ("seed", "fold", "cluster_id"):
        raw = pd.to_numeric(df[col], errors="coerce")
        if raw.isna().any() or not np.all(np.equal(raw, np.floor(raw))):
            raise RuntimeError(f"{path}: invalid or missing {col}; group counts cannot be guessed")
        df[col] = raw.astype(int)
    df = df[df["seed"].isin(selected_seeds)].copy()
    if df.duplicated(["basin_id", "seed"]).any():
        raise RuntimeError(f"{path}: duplicate basin/seed entries")
    expected_set = set(expected_basins)
    if set(df["basin_id"]) != expected_set:
        raise RuntimeError(f"{path}: per-seed metric basin set differs from SHAP performance basin set")
    seed_count_per_basin = df.groupby("basin_id")["seed"].nunique()
    if not (seed_count_per_basin == len(selected_seeds)).all():
        bad = seed_count_per_basin[seed_count_per_basin != len(selected_seeds)]
        raise RuntimeError(f"{path}: not all basins have {len(selected_seeds)} evaluated seeds: {bad.head().to_dict()}")
    if len(df) != len(expected_set) * len(selected_seeds):
        raise RuntimeError(f"{path}: incomplete basin/seed combinations")
    if "fold" in df:
        folds_per_basin = df.groupby("basin_id")["fold"].nunique()
        if (folds_per_basin != 1).any():
            raise RuntimeError(f"{path}: the same basin is evaluated under different PUB folds across seeds")

    cache: dict[tuple[int, int, int], int] = {}
    group_sizes = []
    for row in df.itertuples(index=False):
        key = (int(row.seed), int(row.fold), int(row.cluster_id))
        if key not in cache:
            seed, fold, cluster = key
            if model_name == "cluster":
                model_dir = model_root / f"seed{seed}" / f"fold_{fold}" / "cluster_models" / f"cluster_{cluster}"
            else:
                model_dir = model_root / f"seed{seed}" / f"fold_{fold}" / f"cluster_{cluster}"
            cfg_path = model_dir / "run_cfg.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Missing actual group training count for {key}: {cfg_path}")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            for cfg_key, correct in (("seed", seed), ("fold", fold), ("cluster_id", cluster)):
                if cfg_key in cfg and int(cfg[cfg_key]) != correct:
                    raise RuntimeError(f"{cfg_path}: mismatch in {cfg_key}, expected {correct}")
            expected_init = {"pretrain_finetune": "global", "meta_finetune": "meta"}.get(model_name)
            if expected_init is not None and cfg.get("init_type") != expected_init:
                raise RuntimeError(f"{cfg_path}: expected init_type={expected_init}, found {cfg.get('init_type')}")
            count = cfg.get("n_train_basins")
            training_basins = cfg.get("train_basins")
            if count is None and isinstance(training_basins, list):
                count = len(training_basins)
            if count is None or int(count) <= 0:
                raise RuntimeError(f"{cfg_path}: no valid n_train_basins or train_basins")
            count = int(count)
            if isinstance(training_basins, list):
                if len(set(training_basins)) != count or len(training_basins) != count:
                    raise RuntimeError(f"{cfg_path}: n_train_basins disagrees with train_basins")
            cache[key] = count
        group_sizes.append(cache[key])
    df[GROUP_COUNT_COLUMN] = np.asarray(group_sizes, dtype=np.int64)
    grouping = "mean" if metric_source == "ensemble" else ("median" if metric_source.startswith("median") else "first")
    count_df = df.groupby("basin_id", as_index=False).agg({GROUP_COUNT_COLUMN: grouping})
    count_df[GROUP_COUNT_COLUMN] = count_df[GROUP_COUNT_COLUMN].astype(float)
    print(f"[group-count] {MODEL_LABELS[model_name]} source={metric_source} "
          f"seeds={selected_seeds} experts={len(cache)} mode={grouping} "
          f"range=[{count_df[GROUP_COUNT_COLUMN].min():.1f}, {count_df[GROUP_COUNT_COLUMN].max():.1f}]")
    return count_df, df


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        low = col.lower()
        if low == "nse":
            rename[col] = "nse"
        elif low == "kge":
            rename[col] = "kge"
        elif low == "rmse":
            rename[col] = "rmse"
        elif low == "mae":
            rename[col] = "mae"
        elif low == "fhv":
            rename[col] = "fhv"
        elif low == "flv":
            rename[col] = "flv"
    out = df.rename(columns=rename).copy()
    if "basin_id" not in out.columns and "gauge_id" in out.columns:
        out = out.rename(columns={"gauge_id": "basin_id"})
    if "basin_id" not in out.columns:
        raise KeyError("Metrics file must contain basin_id or gauge_id")
    out["basin_id"] = out["basin_id"].map(normalize_basin_id)
    if "nse" not in out.columns:
        raise KeyError("Metrics file must contain NSE/nse")
    return out


def load_huc_and_coordinates(camels_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_file = camels_root / "camels_attributes_v2.0" / "camels_name.txt"
    topo_file = camels_root / "camels_attributes_v2.0" / "camels_topo.txt"
    if not name_file.exists():
        raise FileNotFoundError(name_file)
    if not topo_file.exists():
        raise FileNotFoundError(topo_file)

    name_df = pd.read_csv(name_file, sep=";", dtype=str)
    name_df.columns = [str(c).strip() for c in name_df.columns]
    name_df["gauge_id"] = name_df["gauge_id"].map(normalize_basin_id)
    name_df["huc_02"] = name_df["huc_02"].astype(str).str.zfill(2)
    keep = [c for c in ["gauge_id", "gauge_name", "huc_02"] if c in name_df.columns]
    name_df = name_df[keep].drop_duplicates("gauge_id")

    topo_df = pd.read_csv(topo_file, sep=";", dtype=str)
    topo_df.columns = [str(c).strip() for c in topo_df.columns]
    topo_df["gauge_id"] = topo_df["gauge_id"].map(normalize_basin_id)
    topo_df["gauge_lat"] = pd.to_numeric(topo_df["gauge_lat"], errors="coerce")
    topo_df["gauge_lon"] = pd.to_numeric(topo_df["gauge_lon"], errors="coerce")
    topo_df = topo_df[["gauge_id", "gauge_lat", "gauge_lon"]].drop_duplicates("gauge_id")
    return name_df, topo_df


def load_static_attributes(
    camels_root: Path,
    basins: list[str],
    cache_db: Path,
) -> pd.DataFrame:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    if not cache_db.exists():
        add_camels_attributes(camels_root, db_path=str(cache_db))
    df = load_attributes(str(cache_db), basins, drop_lat_lon=True).copy()
    df.index = [normalize_basin_id(x) for x in df.index]
    df.index.name = "basin_id"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reindex(basins)
    if df.shape[1] != 27:
        raise RuntimeError(f"Expected 27 static attributes, found {df.shape[1]}: {df.columns.tolist()}")
    return df


def get_shap_values_2d(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP shape: {values.shape}")
    return values


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def run_model(
    model_name: str,
    metrics_df: pd.DataFrame,
    metric_source: str,
    static_df: pd.DataFrame,
    huc_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    output_root: Path,
    group_count_df: pd.DataFrame | None = None,
    group_count_by_seed_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    label = MODEL_LABELS[model_name]
    metrics_df = normalize_metric_columns(metrics_df)
    if metrics_df["basin_id"].duplicated().any():
        examples = metrics_df.loc[metrics_df["basin_id"].duplicated(), "basin_id"].head(5).tolist()
        raise RuntimeError(f"{label}: duplicate basin metrics cannot be silently dropped: {examples}")
    if not np.isfinite(pd.to_numeric(metrics_df["nse"], errors="coerce").to_numpy(dtype=float)).all():
        raise RuntimeError(f"{label}: NSE contains missing/non-finite values; cannot assign performance classes")
    basin_ids = metrics_df["basin_id"].tolist()

    attrs = static_df.reindex(basin_ids)
    analysis = metrics_df.merge(huc_df, left_on="basin_id", right_on="gauge_id", how="left")
    analysis = analysis.drop(columns=["gauge_id"])
    analysis = analysis.merge(coord_df, left_on="basin_id", right_on="gauge_id", how="left")
    analysis = analysis.drop(columns=["gauge_id"])
    analysis = analysis.merge(attrs.reset_index(), on="basin_id", how="left")
    if model_name == "global":
        if group_count_df is not None:
            raise RuntimeError("G model should use only its 27 static attributes")
    else:
        if group_count_df is None or group_count_by_seed_df is None:
            raise RuntimeError(f"{label}: training-group counts must be provided for SHAP")
        analysis = analysis.merge(group_count_df, on="basin_id", how="left", validate="one_to_one")
        if analysis[GROUP_COUNT_COLUMN].isna().any():
            raise RuntimeError(f"{label}: some basins have no training-group count")

    if analysis["huc_02"].isna().any():
        missing = analysis.loc[analysis["huc_02"].isna(), "basin_id"].tolist()
        raise RuntimeError(f"{label}: missing HUC for basins {missing[:10]}")

    if analysis[["gauge_lat", "gauge_lon"]].isna().any().any():
        missing = analysis.loc[
            analysis[["gauge_lat", "gauge_lon"]].isna().any(axis=1), "basin_id"
        ].tolist()
        raise RuntimeError(f"{label}: missing coordinates for basins {missing[:10]}")

    feature_cols = static_df.columns.tolist()
    if model_name != "global":
        feature_cols.append(GROUP_COUNT_COLUMN)
    if analysis[feature_cols].isna().all(axis=0).any():
        bad = analysis[feature_cols].columns[analysis[feature_cols].isna().all(axis=0)].tolist()
        raise RuntimeError(f"{label}: all values missing for attributes {bad}")

    nse_threshold = float(analysis["nse"].median())
    analysis["performance_class_code"] = (analysis["nse"] >= nse_threshold).astype(int)
    analysis["performance_class"] = np.where(
        analysis["performance_class_code"] == 1,
        "high",
        "low",
    )

    X = analysis[feature_cols].copy()
    y = analysis["performance_class_code"].to_numpy(dtype=int)
    huc_values = analysis["huc_02"].astype(str).to_numpy()
    huc_list = sorted(pd.unique(huc_values).tolist())

    if len(huc_list) != 18:
        raise RuntimeError(f"{label}: expected 18 HUC02 regions, found {len(huc_list)}: {huc_list}")

    n_samples = len(analysis)
    n_features = len(feature_cols)
    oof_prob = np.full(n_samples, np.nan, dtype=float)
    oof_pred = np.full(n_samples, -1, dtype=int)
    oof_shap = np.full((n_samples, n_features), np.nan, dtype=float)
    fold_rows = []

    for spatial_fold, held_out_huc in enumerate(huc_list, start=1):
        val_mask = huc_values == held_out_huc
        train_mask = ~val_mask
        X_train = X.loc[train_mask, feature_cols]
        X_val = X.loc[val_mask, feature_cols]
        y_train = y[train_mask]
        y_val = y[val_mask]

        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"{label}: training data have one class when HUC {held_out_huc} is held out")

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        explainer = shap.TreeExplainer(model)
        shap_values = get_shap_values_2d(explainer, X_val)

        oof_prob[val_mask] = prob
        oof_pred[val_mask] = pred
        oof_shap[val_mask, :] = shap_values

        fold_rows.append(
            {
                "spatial_fold": spatial_fold,
                "held_out_huc02": held_out_huc,
                "n_validation_basins": int(val_mask.sum()),
                "n_high_performance": int(y_val.sum()),
                "n_low_performance": int((1 - y_val).sum()),
                "roc_auc": safe_auc(y_val, prob),
                "accuracy": float(accuracy_score(y_val, pred)),
                "macro_f1": float(f1_score(y_val, pred, average="macro")),
            }
        )

        print(
            f"[{label}] {spatial_fold:02d}/18 HUC={held_out_huc} "
            f"n={int(val_mask.sum())} "
            f"AUC={fold_rows[-1]['roc_auc']:.4f} "
            f"Acc={fold_rows[-1]['accuracy']:.4f} "
            f"F1={fold_rows[-1]['macro_f1']:.4f}"
        )

    if np.isnan(oof_prob).any() or (oof_pred < 0).any() or np.isnan(oof_shap).any():
        raise RuntimeError(f"{label}: incomplete OOF predictions or SHAP values")

    overall_auc = float(roc_auc_score(y, oof_prob))
    overall_acc = float(accuracy_score(y, oof_pred))
    overall_f1 = float(f1_score(y, oof_pred, average="macro"))
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])

    model_out = output_root / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    spatial_cols = [
        c
        for c in [
            "basin_id",
            "gauge_name",
            "huc_02",
            "gauge_lat",
            "gauge_lon",
            "fold",
            "nse",
            "kge",
            "rmse",
            "mae",
            "fhv",
            "flv",
            "performance_class_code",
            "performance_class",
            GROUP_COUNT_COLUMN,
        ]
        if c in analysis.columns
    ]
    spatial_df = analysis[spatial_cols].copy()
    spatial_df["oof_high_performance_probability"] = oof_prob
    spatial_df["oof_predicted_class_code"] = oof_pred
    spatial_df["oof_predicted_class"] = np.where(oof_pred == 1, "high", "low")
    spatial_df["oof_correct"] = (y == oof_pred).astype(int)
    spatial_df.to_csv(model_out / "spatial_oof_results.csv", index=False, encoding="utf-8-sig")
    if group_count_by_seed_df is not None:
        group_count_by_seed_df.to_csv(
            model_out / "group_training_basin_count_by_seed.csv", index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(fold_rows).to_csv(
        model_out / "spatial_cv_by_huc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.mean(np.abs(oof_shap), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(model_out / "shap_importance.csv", index=False, encoding="utf-8-sig")

    long_frames = []
    for j, feature in enumerate(feature_cols):
        long_frames.append(
            pd.DataFrame(
                {
                    "basin_id": analysis["basin_id"].to_numpy(),
                    "huc_02": analysis["huc_02"].to_numpy(),
                    "nse": analysis["nse"].to_numpy(dtype=float),
                    "performance_class": analysis["performance_class"].to_numpy(),
                    "feature": feature,
                    "feature_value": analysis[feature].to_numpy(dtype=float),
                    "oof_shap_value": oof_shap[:, j],
                }
            )
        )
    shap_long = pd.concat(long_frames, ignore_index=True)
    shap_long.to_csv(model_out / "shap_long.csv", index=False, encoding="utf-8-sig")

    top10 = importance.head(10)["feature"].tolist()
    shap_long[shap_long["feature"].isin(top10)].to_csv(
        model_out / "shap_top10_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance.head(10).to_csv(
        model_out / "shap_top10_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model": label,
        "model_key": model_name,
        "metric_source": metric_source,
        "n_basins": int(len(analysis)),
        "n_huc02": int(len(huc_list)),
        "n_static_attributes": int(static_df.shape[1]),
        "n_shap_features": int(len(feature_cols)),
        "group_training_count_feature": GROUP_COUNT_COLUMN if model_name != "global" else None,
        "group_training_count_source": (
            "expert's n_train_basins" if model_name == "cluster" else
            "cluster finetuning's n_train_basins" if model_name != "global" else None
        ),
        "nse_median_threshold": nse_threshold,
        "oof_roc_auc": overall_auc,
        "oof_accuracy": overall_acc,
        "oof_macro_f1": overall_f1,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "xgboost_parameters": XGB_PARAMS,
        "shap_protocol": "Leave-One-HUC02-Out out-of-fold TreeSHAP",
    }
    (model_out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[{label}] overall AUC={overall_auc:.6f} "
        f"Accuracy={overall_acc:.6f} Macro-F1={overall_f1:.6f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--camels_root", type=Path, default=DEFAULT_CAMELS_ROOT)
    parser.add_argument("--evaluation_root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--st_root", type=Path, default=DEFAULT_ST_ROOT)
    parser.add_argument("--ptst_root", type=Path, default=DEFAULT_PTST_ROOT)
    parser.add_argument("--mst_root", type=Path, default=DEFAULT_MST_ROOT)
    parser.add_argument("--ensemble_seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--performance_source",
        type=str,
        default="ensemble",
        help="ensemble, median, or seed<seed-number> (e.g. seed12345)",
    )
    parser.add_argument("--allow_non531", action="store_true")
    args = parser.parse_args()

    huc_df, coord_df = load_huc_and_coordinates(args.camels_root)
    cache_db = args.output_root / "_cache" / "camels_attributes.db"

    metric_payload = {}
    union_basins = set()
    for model_name in args.models:
        metrics_df, source = load_metrics(model_name, args.evaluation_root, args.performance_source)
        metrics_df = normalize_metric_columns(metrics_df)
        metric_payload[model_name] = (metrics_df, source)
        union_basins.update(metrics_df["basin_id"].tolist())

    union_basins = sorted(normalize_basin_id(x) for x in union_basins)
    static_df = load_static_attributes(args.camels_root, union_basins, cache_db)

    summaries = []
    for model_name in args.models:
        metrics_df, source = metric_payload[model_name]
        if not args.allow_non531 and metrics_df["basin_id"].nunique() != 531:
            raise RuntimeError(
                f"{MODEL_LABELS[model_name]}: expected 531 basins, found {metrics_df['basin_id'].nunique()}"
            )
        group_count_df, group_count_by_seed_df = (None, None)
        if model_name != "global":
            group_count_df, group_count_by_seed_df = load_group_training_counts(
                model_name=model_name,
                evaluation_root=args.evaluation_root,
                metric_source=source,
                expected_basins=metrics_df["basin_id"].tolist(),
                st_root=args.st_root,
                ptst_root=args.ptst_root,
                mst_root=args.mst_root,
                ensemble_seeds=args.ensemble_seeds,
            )
        summary = run_model(
            model_name=model_name,
            metrics_df=metrics_df,
            metric_source=source,
            static_df=static_df,
            huc_df=huc_df,
            coord_df=coord_df,
            output_root=args.output_root,
            group_count_df=group_count_df,
            group_count_by_seed_df=group_count_by_seed_df,
        )
        summaries.append(summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        args.output_root / "model_spatial_shap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[done] {args.output_root}")


if __name__ == "__main__":
    main()
