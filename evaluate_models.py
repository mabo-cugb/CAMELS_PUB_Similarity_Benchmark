from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

import src.datautils as datautils_module
from src.datautils import load_attributes
from src.datasets import CamelsTXT
from src.lstm import LSTM
from src.metrics import calc_fdc_fhv, calc_fdc_flv, calc_nse

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMELS_ROOT = ROOT / "data" / "camels_root"
DEFAULT_SPLIT_ROOT = ROOT / "data" / "splits"
DEFAULT_CLUSTER_INFO_ROOT = ROOT / "data" / "clustering"
DEFAULT_GLOBAL_ROOT = ROOT / "outputs" / "global_lstm"
DEFAULT_CLUSTER_ROOT = ROOT / "outputs" / "cluster_lstm"
DEFAULT_PT_ROOT = ROOT / "outputs" / "pretrain_finetune_lstm"
DEFAULT_META_ROOT = ROOT / "outputs" / "meta_pretrain"
DEFAULT_MST_ROOT = ROOT / "outputs" / "meta_finetune_lstm"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "evaluation"

TEST_START = "1999-10-01"
TEST_END = "2004-09-30"
SEQ_LENGTH = 270
DYN_INPUT_SIZE = 5
FORGET_GATE_BIAS = 5
SEEDS = [12345, 1234, 123]
N_FOLDS = 5
MODEL_ORDER = ["global", "cluster", "pretrain_finetune", "meta_finetune"]
MODEL_LABELS = {
    "global": "G",
    "cluster": "ST",
    "pretrain_finetune": "PT-ST",
    "meta_finetune": "MST",
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_basin_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if not digits:
        raise ValueError(f"Invalid basin id: {value!r}")
    return digits.zfill(8)


def read_basin_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            values.append(normalize_basin_id(line))
    return list(dict.fromkeys(values))


def resolve_split_dir(split_root: Path, fold: int) -> Path:
    candidates = [split_root / f"fold_{fold}" / "split", split_root / f"fold_{fold}"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing fold_{fold} split under {split_root}")


def load_splits(split_root: Path) -> dict[int, tuple[list[str], list[str]]]:
    folds = {}
    all_test = []
    for fold in range(1, N_FOLDS + 1):
        split_dir = resolve_split_dir(split_root, fold)
        gauged = read_basin_list(split_dir / "gauged_basins.txt")
        ungauged = read_basin_list(split_dir / "ungauged_basins.txt")
        if set(gauged) & set(ungauged):
            raise RuntimeError(f"fold_{fold}: gauged and ungauged overlap")
        folds[fold] = (gauged, ungauged)
        all_test.extend(ungauged)
    if len(all_test) != len(set(all_test)):
        raise RuntimeError("A PUB test basin appears in more than one fold")
    return folds


def find_cluster_mapping(cluster_info_root: Path, seed: int, fold: int) -> Path | None:
    candidates = [
        cluster_info_root / f"seed{seed}" / f"fold_{fold}" / "ungauged_pred_clusters.csv",
        cluster_info_root / f"seed{seed}" / f"fold_{fold}" / "clustering" / "ungauged_pred_clusters.csv",
        cluster_info_root / f"fold_{fold}" / "ungauged_pred_clusters.csv",
        cluster_info_root / f"fold_{fold}" / "clustering" / "ungauged_pred_clusters.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_cluster_info_pickle(cluster_info_root: Path, seed: int, fold: int) -> Path | None:
    candidates = [
        cluster_info_root / f"seed{seed}" / f"fold_{fold}" / "cluster_info.pkl",
        cluster_info_root / f"seed{seed}" / f"fold_{fold}" / "clustering" / "cluster_info.pkl",
        cluster_info_root / f"fold_{fold}" / "cluster_info.pkl",
        cluster_info_root / f"fold_{fold}" / "clustering" / "cluster_info.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_ungauged_mapping(cluster_info_root: Path, seed: int, fold: int, ungauged: list[str]) -> dict[str, int]:
    csv_path = find_cluster_mapping(cluster_info_root, seed, fold)
    mapping: dict[str, int] = {}
    if csv_path is not None:
        df = pd.read_csv(csv_path, dtype=str)
        id_col = next((c for c in ["basin_id", "gauge_id", "basin"] if c in df.columns), None)
        cluster_col = next((c for c in ["pred_cluster", "cluster_id", "cluster"] if c in df.columns), None)
        if id_col is None or cluster_col is None:
            raise RuntimeError(f"{csv_path} must contain basin id and predicted cluster columns")
        for _, row in df.iterrows():
            mapping[normalize_basin_id(row[id_col])] = int(float(row[cluster_col]))
    else:
        pkl_path = find_cluster_info_pickle(cluster_info_root, seed, fold)
        if pkl_path is None:
            raise FileNotFoundError(
                f"Missing ungauged cluster mapping for seed={seed}, fold={fold} under {cluster_info_root}"
            )
        with pkl_path.open("rb") as f:
            info = pickle.load(f)
        details = info.get("cluster_details", {})
        for key, value in details.items():
            cid = int(key)
            if not isinstance(value, dict):
                continue
            for basin in value.get("basins_ungauged", []):
                mapping[normalize_basin_id(basin)] = cid
    missing = sorted(set(ungauged) - set(mapping))
    if missing:
        raise RuntimeError(
            f"seed={seed} fold={fold}: cluster mapping misses {len(missing)} test basins, example={missing[:5]}"
        )
    return {basin: int(mapping[basin]) for basin in ungauged}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_dynamic_output_scaler(path: Path) -> dict[str, Any]:
    scaler = load_json(path)
    required = ["input_means", "input_stds", "output_mean", "output_std"]
    missing = [key for key in required if key not in scaler]
    if missing:
        raise RuntimeError(f"{path} is missing scaler fields: {missing}")
    input_means = np.asarray(scaler["input_means"], dtype=np.float64)
    input_stds = np.asarray(scaler["input_stds"], dtype=np.float64)
    output_mean = float(scaler["output_mean"])
    output_std = float(scaler["output_std"])
    if input_means.shape != (DYN_INPUT_SIZE,) or input_stds.shape != (DYN_INPUT_SIZE,):
        raise RuntimeError(f"{path} has invalid dynamic-input scaler shape")
    if np.any(~np.isfinite(input_means)) or np.any(~np.isfinite(input_stds)) or np.any(input_stds <= 0.0):
        raise RuntimeError(f"{path} has invalid dynamic-input scaler values")
    if not np.isfinite(output_mean) or not np.isfinite(output_std) or output_std <= 0.0:
        raise RuntimeError(f"{path} has invalid output scaler values")
    return scaler


def install_scaler(scaler: dict[str, Any]) -> None:
    datautils_module.SCALER = {
        "input_means": np.asarray(scaler["input_means"], dtype=np.float64),
        "input_stds": np.asarray(scaler["input_stds"], dtype=np.float64),
        "output_mean": np.asarray([scaler["output_mean"]], dtype=np.float64),
        "output_std": np.asarray([scaler["output_std"]], dtype=np.float64),
    }


def load_static_norm(path: Path) -> tuple[pd.Series, pd.Series, list[str]]:
    cfg = load_json(path)
    columns = list(cfg["columns"])
    means = pd.Series({k: float(v) for k, v in cfg["means"].items()}).reindex(columns).fillna(0.0)
    stds = pd.Series({k: float(v) for k, v in cfg["stds"].items()}).reindex(columns).replace(0.0, 1.0).fillna(1.0)
    return means, stds, columns


class RegionalLSTM(nn.Module):
    def __init__(self, input_size_stat: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = LSTM(
            input_size=DYN_INPUT_SIZE + input_size_stat,
            hidden_size=hidden_size,
            batch_first=True,
            initial_forget_bias=FORGET_GATE_BIAS,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        if x_s.dim() == 3:
            x_s = x_s.squeeze(1)
        steps = x_d.shape[1]
        x_s_rep = x_s.unsqueeze(1).expand(-1, steps, -1)
        x = torch.cat([x_d, x_s_rep], dim=-1)
        h_seq, _ = self.lstm(x)
        return self.fc(self.dropout(h_seq[:, -1, :]))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def resolve_artifacts(
    model_name: str,
    seed: int,
    fold: int,
    cluster_id: int | None,
    global_root: Path,
    cluster_root: Path,
    pt_root: Path,
    meta_root: Path,
    mst_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    if model_name == "global":
        model_dir = global_root / f"seed{seed}" / f"fold_{fold}"
        model_path = model_dir / "model_best.pt"
        cfg_path = model_dir / "run_cfg.json"
        norm_path = model_dir / "static_norm.json"
        db_path = global_root / "shared_data" / f"fold_{fold}" / "attributes_filtered.db"
        scaler_path = model_dir / "dynamic_output_scaler.json"
    elif model_name == "cluster":
        if cluster_id is None:
            raise ValueError("cluster_id is required")
        model_dir = cluster_root / f"seed{seed}" / f"fold_{fold}" / "cluster_models" / f"cluster_{cluster_id}"
        model_path = model_dir / "model_best.pt"
        cfg_path = model_dir / "run_cfg.json"
        norm_path = model_dir / "static_norm.json"
        db_path = cluster_root / "shared_data" / f"fold_{fold}" / "attributes_filtered.db"
        scaler_path = model_dir / "dynamic_output_scaler.json"
    elif model_name == "pretrain_finetune":
        if cluster_id is None:
            raise ValueError("cluster_id is required")
        model_dir = pt_root / f"seed{seed}" / f"fold_{fold}" / f"cluster_{cluster_id}"
        model_path = model_dir / "model_best.pt"
        cfg_path = model_dir / "run_cfg.json"
        norm_path = global_root / f"seed{seed}" / f"fold_{fold}" / "static_norm.json"
        db_path = global_root / "shared_data" / f"fold_{fold}" / "attributes_filtered.db"
        scaler_path = model_dir / "dynamic_output_scaler.json"
    elif model_name == "meta_finetune":
        if cluster_id is None:
            raise ValueError("cluster_id is required")
        model_dir = mst_root / f"seed{seed}" / f"fold_{fold}" / f"cluster_{cluster_id}"
        model_path = model_dir / "model_best.pt"
        cfg_path = model_dir / "run_cfg.json"
        norm_path = meta_root / f"seed{seed}" / f"fold_{fold}" / "static_norm.json"
        db_path = meta_root / "shared_data" / f"fold_{fold}" / "attributes_filtered.db"
        scaler_path = model_dir / "dynamic_output_scaler.json"
    else:
        raise ValueError(model_name)
    for path in [model_path, cfg_path, norm_path, db_path, scaler_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    return model_path, cfg_path, norm_path, db_path, scaler_path

def load_static_vector(db_path: Path, basin: str, columns: list[str], means: pd.Series, stds: pd.Series) -> torch.Tensor:
    df = load_attributes(str(db_path), [basin], drop_lat_lon=True)
    df.index = [normalize_basin_id(x) for x in df.index]
    if basin not in df.index:
        raise RuntimeError(f"Basin {basin} missing from attributes database {db_path}")
    row = df.loc[[basin]].reindex(columns=columns)
    if row.isna().any().any():
        bad = row.columns[row.isna().any()].tolist()
        raise RuntimeError(f"Basin {basin} has missing static attributes: {bad}")
    values = ((row.iloc[0] - means) / stds).to_numpy(dtype=np.float32)
    return torch.from_numpy(values)


def calc_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float).reshape(-1)
    if len(obs) < 2 or np.std(obs) == 0 or np.mean(obs) == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    if not np.isfinite(r):
        return float("nan")
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def calculate_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float).reshape(-1)
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0)
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) < 2:
        return {"nse": np.nan, "kge": np.nan, "rmse": np.nan, "mae": np.nan, "fhv": np.nan, "flv": np.nan, "n_days": len(obs)}
    try:
        nse = float(calc_nse(obs.copy(), sim.copy()))
    except Exception:
        nse = np.nan
    kge = calc_kge(obs, sim)
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    mae = float(np.mean(np.abs(sim - obs)))
    try:
        fhv = float(calc_fdc_fhv(obs.copy(), sim.copy()))
    except Exception:
        fhv = np.nan
    try:
        flv = float(calc_fdc_flv(obs.copy(), sim.copy()))
    except Exception:
        flv = np.nan
    return {"nse": nse, "kge": kge, "rmse": rmse, "mae": mae, "fhv": fhv, "flv": flv, "n_days": int(len(obs))}


def predict_basin(
    camels_root: Path,
    basin: str,
    model_path: Path,
    cfg_path: Path,
    norm_path: Path,
    db_path: Path,
    scaler_path: Path,
    batch_size: int,
    num_workers: int,
    clip_negative: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = load_dynamic_output_scaler(scaler_path)
    install_scaler(scaler)
    cfg = load_json(cfg_path)
    hidden_size = int(cfg["hidden_size"])
    dropout = float(cfg.get("dropout", 0.4))
    means, stds, columns = load_static_norm(norm_path)
    model = RegionalLSTM(len(columns), hidden_size, dropout).to(DEVICE)
    model.load_state_dict(load_state(model_path), strict=True)
    model.eval()
    dataset = CamelsTXT(
        camels_root=camels_root,
        basin=basin,
        dates=[pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)],
        is_train=False,
        seq_length=SEQ_LENGTH,
        with_attributes=False,
    )
    static = load_static_vector(db_path, basin, columns, means, stds)
    loader = DataLoader(TensorDataset(dataset.x, dataset.y), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds = []
    obs = []
    with torch.no_grad():
        for x_d, y in loader:
            x_d = x_d.to(DEVICE, non_blocking=True)
            x_s = static.unsqueeze(0).expand(x_d.shape[0], -1).to(DEVICE, non_blocking=True)
            pred = model(x_d, x_s).detach().cpu().numpy()
            preds.append(pred)
            obs.append(y.numpy())
    pred_norm = np.concatenate(preds, axis=0).reshape(-1, 1)
    sim = datautils_module.rescale_features(pred_norm, variable="output").reshape(-1)
    obs_arr = np.concatenate(obs, axis=0).reshape(-1)
    if clip_negative:
        sim = np.maximum(sim, 0.0)
    dates = pd.date_range(TEST_START, TEST_END, freq="D")
    if len(dates) != len(obs_arr):
        dates = dates[: len(obs_arr)]
    return obs_arr, sim, dates.to_numpy()


def summarize_metrics(df: pd.DataFrame, model_label: str, seed_label: str) -> dict[str, Any]:
    valid_nse = df["nse"].dropna()
    valid_kge = df["kge"].dropna()
    return {
        "model": model_label,
        "seed": seed_label,
        "n_basins": int(len(df)),
        "median_nse": float(valid_nse.median()) if len(valid_nse) else np.nan,
        "mean_nse": float(valid_nse.mean()) if len(valid_nse) else np.nan,
        "median_kge": float(valid_kge.median()) if len(valid_kge) else np.nan,
        "mean_kge": float(valid_kge.mean()) if len(valid_kge) else np.nan,
        "nse_gt_0": int((valid_nse > 0).sum()),
        "nse_gt_0_5": int((valid_nse > 0.5).sum()),
        "kge_gt_0_5": int((valid_kge > 0.5).sum()),
        "negative_nse": int((valid_nse < 0).sum()),
        "median_rmse": float(df["rmse"].median()),
        "median_mae": float(df["mae"].median()),
        "median_fhv": float(df["fhv"].median()),
        "median_flv": float(df["flv"].median()),
    }


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom > 0 else 0.0


def evaluate_one_model(
    model_name: str,
    seeds: list[int],
    folds: list[int],
    splits: dict[int, tuple[list[str], list[str]]],
    camels_root: Path,
    cluster_info_root: Path,
    global_root: Path,
    cluster_root: Path,
    pt_root: Path,
    meta_root: Path,
    mst_root: Path,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    clip_negative: bool,
    save_timeseries: bool,
    allow_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_label = MODEL_LABELS[model_name]
    model_out = output_root / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    rows = []
    ensemble_buffer: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_rows = []
        for fold in folds:
            _, ungauged = splits[fold]
            mapping = None
            if model_name != "global":
                mapping = load_ungauged_mapping(cluster_info_root, seed, fold, ungauged)
            for i, basin in enumerate(ungauged, 1):
                cluster_id = None if mapping is None else int(mapping[basin])
                try:
                    model_path, cfg_path, norm_path, db_path, scaler_path = resolve_artifacts(
                        model_name,
                        seed,
                        fold,
                        cluster_id,
                        global_root,
                        cluster_root,
                        pt_root,
                        meta_root,
                        mst_root,
                    )
                    obs, sim, dates = predict_basin(
                        camels_root,
                        basin,
                        model_path,
                        cfg_path,
                        norm_path,
                        db_path,
                        scaler_path,
                        batch_size,
                        num_workers,
                        clip_negative,
                    )
                except Exception as exc:
                    if not allow_missing:
                        raise
                    print(f"[skip] {model_label} seed={seed} fold={fold} basin={basin}: {exc}")
                    continue
                metrics = calculate_metrics(obs, sim)
                row = {
                    "model": model_label,
                    "model_key": model_name,
                    "seed": seed,
                    "fold": fold,
                    "basin_id": basin,
                    "cluster_id": cluster_id,
                    **metrics,
                }
                rows.append(row)
                seed_rows.append(row)
                entry = ensemble_buffer.setdefault(basin, {"obs": obs, "dates": dates, "sims": [], "fold": fold, "clusters": []})
                if len(entry["obs"]) != len(obs) or not np.allclose(entry["obs"], obs, equal_nan=True):
                    raise RuntimeError(f"Observed series mismatch across seeds for basin {basin}")
                entry["sims"].append(sim)
                entry["clusters"].append(cluster_id)
                if save_timeseries:
                    ts_dir = model_out / "timeseries" / f"seed{seed}" / f"fold_{fold}"
                    ts_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"date": dates, "qobs": obs, "qsim": sim}).to_csv(ts_dir / f"{basin}.csv", index=False)
                if i == 1 or i % 25 == 0 or i == len(ungauged):
                    print(f"[{model_label}] seed={seed} fold={fold} {i}/{len(ungauged)}")
        seed_df = pd.DataFrame(seed_rows)
        seed_df.to_csv(model_out / f"basin_metrics_seed{seed}.csv", index=False, encoding="utf-8-sig")
    all_df = pd.DataFrame(rows)
    all_df.to_csv(model_out / "basin_metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    ensemble_rows = []
    if len(seeds) > 1:
        for basin, entry in ensemble_buffer.items():
            sims = entry["sims"]
            if len(sims) != len(seeds):
                if allow_missing:
                    continue
                raise RuntimeError(f"Basin {basin} has {len(sims)} seed predictions, expected {len(seeds)}")
            sim_ens = np.mean(np.stack(sims, axis=0), axis=0)
            metrics = calculate_metrics(entry["obs"], sim_ens)
            ensemble_rows.append(
                {
                    "model": model_label,
                    "model_key": model_name,
                    "seed": "ensemble",
                    "fold": entry["fold"],
                    "basin_id": basin,
                    "cluster_id": np.nan,
                    **metrics,
                }
            )
            if save_timeseries:
                ens_dir = model_out / "timeseries" / "ensemble"
                ens_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"date": entry["dates"], "qobs": entry["obs"], "qsim": sim_ens}).to_csv(ens_dir / f"{basin}.csv", index=False)
    ensemble_df = pd.DataFrame(ensemble_rows)
    if not ensemble_df.empty:
        ensemble_df.to_csv(model_out / "basin_metrics_ensemble.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    if not all_df.empty:
        for seed in seeds:
            part = all_df[all_df["seed"] == seed]
            if not part.empty:
                summary_rows.append(summarize_metrics(part, model_label, str(seed)))
    if not ensemble_df.empty:
        summary_rows.append(summarize_metrics(ensemble_df, model_label, "ensemble"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(model_out / "summary.csv", index=False, encoding="utf-8-sig")
    return all_df, ensemble_df, summary_df


def build_pairwise_comparisons(ensemble_by_model: dict[str, pd.DataFrame], output_root: Path) -> pd.DataFrame:
    rows = []
    available = [key for key in MODEL_ORDER if key in ensemble_by_model and not ensemble_by_model[key].empty]
    for a, b in combinations(available, 2):
        left = ensemble_by_model[a][["basin_id", "nse", "kge"]].rename(columns={"nse": "nse_a", "kge": "kge_a"})
        right = ensemble_by_model[b][["basin_id", "nse", "kge"]].rename(columns={"nse": "nse_b", "kge": "kge_b"})
        merged = left.merge(right, on="basin_id", how="inner")
        for metric in ["nse", "kge"]:
            x = merged[f"{metric}_a"].to_numpy(dtype=float)
            y = merged[f"{metric}_b"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) == 0:
                continue
            try:
                stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                stat = float(stat)
                p_value = float(p_value)
            except Exception:
                stat = np.nan
                p_value = np.nan
            rows.append(
                {
                    "model_a": MODEL_LABELS[a],
                    "model_b": MODEL_LABELS[b],
                    "metric": metric.upper(),
                    "n_basins": int(len(x)),
                    "median_a": float(np.median(x)),
                    "median_b": float(np.median(y)),
                    "median_difference_a_minus_b": float(np.median(x - y)),
                    "a_better": int(np.sum(x > y)),
                    "b_better": int(np.sum(y > x)),
                    "ties": int(np.sum(x == y)),
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "rank_biserial_a_minus_b": rank_biserial_paired(x, y),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_root / "paired_model_comparisons_ensemble.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, N_FOLDS + 1)))
    parser.add_argument("--camels_root", type=Path, default=DEFAULT_CAMELS_ROOT)
    parser.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cluster_info_root", type=Path, default=DEFAULT_CLUSTER_INFO_ROOT)
    parser.add_argument("--global_root", type=Path, default=DEFAULT_GLOBAL_ROOT)
    parser.add_argument("--cluster_root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--pt_root", type=Path, default=DEFAULT_PT_ROOT)
    parser.add_argument("--meta_root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--mst_root", type=Path, default=DEFAULT_MST_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clip_negative", action="store_true")
    parser.add_argument("--save_timeseries", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.split_root)
    print(f"[device] {DEVICE}")
    print(f"[test] {TEST_START} -> {TEST_END}")
    print(f"[models] {[MODEL_LABELS[x] for x in args.models]}")
    print(f"[seeds] {args.seeds}")
    print(f"[folds] {args.folds}")
    print(f"[clip_negative] {args.clip_negative}")
    all_summaries = []
    ensemble_by_model: dict[str, pd.DataFrame] = {}
    for model_name in args.models:
        print("\n" + "=" * 90)
        print(f"EVALUATE {MODEL_LABELS[model_name]}")
        print("=" * 90)
        _, ensemble_df, summary_df = evaluate_one_model(
            model_name,
            args.seeds,
            args.folds,
            splits,
            args.camels_root,
            args.cluster_info_root,
            args.global_root,
            args.cluster_root,
            args.pt_root,
            args.meta_root,
            args.mst_root,
            args.output_root,
            args.batch_size,
            args.num_workers,
            args.clip_negative,
            args.save_timeseries,
            args.allow_missing,
        )
        if not summary_df.empty:
            all_summaries.append(summary_df)
        if not ensemble_df.empty:
            ensemble_by_model[model_name] = ensemble_df
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(args.output_root / "model_summary.csv", index=False, encoding="utf-8-sig")
    if len(ensemble_by_model) >= 2:
        build_pairwise_comparisons(ensemble_by_model, args.output_root)
    print(f"\n[done] {args.output_root}")


if __name__ == "__main__":
    main()
