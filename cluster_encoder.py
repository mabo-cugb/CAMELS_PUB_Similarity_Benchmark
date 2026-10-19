"""Supervised encoder grouping for CAMELS PUB (one-file public-code addition).

Stages:
  train   Fit a runoff-supervised meteorological LSTM + static MLP encoder on
          GAUGED basins and training-period samples only.
  cluster Extract 16-D mean embeddings from meteorology + static attributes,
          fit feature scaler and KMeans on GAUGED basins ONLY, then assign
          UNGAUGED basins without using their observed streamflow.
  all     Execute both stages.

This file intentionally does NOT train the group-specific LSTM experts: use
train_cluster_lstm.py with --cluster_root pointing to this script's output.
It reuses src/ plus the fold-safe data preparation in train_global_lstm.py.

Example (fixed 513/18 experiment-1 split, K=6):
  python -u cluster_encoder.py --folds 1 --split_root data/exp1_splits --k 6 --output_root outputs/exp1_encoder_clustering
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

import src.datautils as datautils
from src.datautils import load_attributes, load_forcing
from src.datasets import CamelsH5
from src.nseloss import NSELoss
import train_global_lstm as global_helpers

ROOT = Path(__file__).resolve().parent
DYNAMIC = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]
TRAIN_START = "1989-10-01"
TRAIN_END = "1999-09-30"


def basin_id(x: str) -> str:
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid basin ID: {x!r}")
    return text.zfill(8)


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [basin_id(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not lines or len(lines) != len(set(lines)):
        raise ValueError(f"Empty or duplicate basin IDs: {path}")
    return lines


def get_split(split_root: Path, fold: int) -> tuple[list[str], list[str]]:
    for directory in [split_root / f"fold_{fold}" / "split", split_root / f"fold_{fold}"]:
        if (directory / "gauged_basins.txt").exists() and (directory / "ungauged_basins.txt").exists():
            train = read_ids(directory / "gauged_basins.txt")
            test = read_ids(directory / "ungauged_basins.txt")
            if set(train) & set(test):
                raise RuntimeError(f"fold_{fold}: training and held-out basins overlap")
            return train, test
    raise FileNotFoundError(f"fold_{fold} gauged/ungauged lists not found under {split_root}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def signature(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StreamflowEncoder(nn.Module):
    """Architecture from the supplied train_pretrain_encoder.py (128 -> 16)."""

    def __init__(self, static_dim: int, hidden: int, encoding: int, dropout: float, forget_bias: float):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden, num_layers=1, batch_first=True, dropout=0.0)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, encoding),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(encoding, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )
        # Retain the original encoder's forget-gate initialization.
        for name, parameter in self.lstm.named_parameters():
            if "bias" in name:
                n = parameter.numel()
                with torch.no_grad():
                    parameter[n // 4:n // 2].fill_(forget_bias)

    def encode_only(self, x_dynamic: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        if x_static.ndim == 3:
            x_static = x_static.squeeze(1)
        h_seq, _ = self.lstm(x_dynamic)
        return self.fusion(torch.cat([h_seq[:, -1, :], self.static_encoder(x_static)], dim=1))

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(self.encode_only(x_dynamic, x_static))


def load_train_statics(filtered_db: Path, train_ids: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = load_attributes(str(filtered_db), train_ids, drop_lat_lon=True)
    df.index = [basin_id(v) for v in df.index]
    if set(df.index) != set(train_ids):
        raise RuntimeError("Missing train basin static attributes")
    df = df.loc[train_ids].apply(pd.to_numeric, errors="raise")
    if df.isna().any().any():
        raise RuntimeError("NaN in training static attributes")
    means = df.mean()
    stds = df.std().replace(0.0, 1.0).fillna(1.0)
    return df, means, stds


def load_all_statics(full_db: Path, ids: list[str], columns: list[str]) -> pd.DataFrame:
    df = load_attributes(str(full_db), ids, drop_lat_lon=True)
    df.index = [basin_id(v) for v in df.index]
    missing = sorted(set(ids) - set(df.index))
    if missing:
        raise RuntimeError(f"Missing static attributes for basins: {missing[:5]}")
    df = df.loc[ids].reindex(columns=columns).apply(pd.to_numeric, errors="raise")
    if df.isna().any().any():
        raise RuntimeError("Missing values in selected static attributes")
    return df


def create_model(args: argparse.Namespace, static_dim: int, device: torch.device) -> StreamflowEncoder:
    return StreamflowEncoder(static_dim, args.hidden_size, args.encoding_dim, args.dropout,
                             args.forget_bias).to(device)


def train_encoder(args: argparse.Namespace, fold: int, train: list[str], filtered_db: Path,
                  h5_path: Path, scaler: dict, device: torch.device) -> tuple[Path, Path]:
    out = args.output_root / f"fold_{fold}" / "encoder"
    out.mkdir(parents=True, exist_ok=True)
    df, means, stds = load_train_statics(filtered_db, train)
    config = {
        "train_basins": train, "train_start": TRAIN_START, "train_end": TRAIN_END,
        "seq_length": args.seq_length, "hidden_size": args.hidden_size,
        "encoding_dim": args.encoding_dim, "dropout": args.dropout,
        "forget_bias": args.forget_bias, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "samples_per_epoch": args.samples_per_epoch,
        "encoder_seed": args.encoder_seed, "dynamic_scaler_hash": signature(scaler),
        "static_columns": list(df.columns), "algorithm": "runoff_supervised_encoder_NSELoss",
    }
    cfg_file = out / "run_cfg.json"
    if cfg_file.exists() and read_json(cfg_file) != config:
        raise RuntimeError(f"Encoder configuration changed: {cfg_file}; use a NEW --output_root")
    if not cfg_file.exists() and any((out / name).exists() for name in
                                     ["checkpoint_latest.pt", "model_best.pt", "TRAIN_DONE.json"]):
        raise RuntimeError(f"Orphan encoder checkpoint without configuration: {out}")
    write_json(cfg_file, config)
    norm_file = out / "static_norm.json"
    static_cfg = {"columns": list(df.columns), "means": means.to_dict(), "stds": stds.to_dict(),
                  "train_basins": train}
    if norm_file.exists() and read_json(norm_file) != static_cfg:
        raise RuntimeError("Existing encoder static normalization differs")
    write_json(norm_file, static_cfg)
    best = out / "model_best.pt"
    if (out / "TRAIN_DONE.json").exists():
        if not best.exists():
            raise RuntimeError("Encoder marked complete but best weights are missing")
        print(f"[encoder skip] fold={fold} already complete")
        return best, norm_file

    datautils.SCALER = {"input_means": np.asarray(scaler["input_means"]),
                        "input_stds": np.asarray(scaler["input_stds"]),
                        "output_mean": np.asarray([scaler["output_mean"]]),
                        "output_std": np.asarray([scaler["output_std"]])}
    ds = CamelsH5(h5_file=h5_path, basins=train, db_path=str(filtered_db),
                  concat_static=False, cache=False, no_static=False)
    ds.df = (df - means) / stds
    ds.attribute_means = means
    ds.attribute_stds = stds
    ds.attribute_names = ds.df.columns
    model = create_model(args, len(df.columns), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = NSELoss()
    start_epoch = 1
    best_loss = float("inf")
    latest = out / "checkpoint_latest.pt"
    if latest.exists():
        try:
            cp = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            cp = torch.load(latest, map_location="cpu")
        if cp.get("config_hash") != signature(config):
            raise RuntimeError("Existing encoder checkpoint belongs to a different configuration")
        model.load_state_dict(cp["model_state"], strict=True)
        optimizer.load_state_dict(cp["optim_state"])
        for s in optimizer.state.values():
            for k, val in s.items():
                if torch.is_tensor(val):
                    s[k] = val.to(device)
        start_epoch = int(cp["epoch"]) + 1
        best_loss = float(cp["best_loss"])
        print(f"[encoder resume] fold={fold} epoch={start_epoch}")
    elif best.exists():
        raise RuntimeError(f"Orphan best encoder weights: {best}")

    set_seed(args.encoder_seed + fold * 1000)
    for epoch in range(start_epoch, args.epochs + 1):
        if args.samples_per_epoch:
            generator = np.random.default_rng(args.encoder_seed + fold * 1000 + epoch)
            indices = generator.integers(0, len(ds), size=args.samples_per_epoch).tolist()
            epoch_ds = torch.utils.data.Subset(ds, indices)
        else:
            epoch_ds = ds
        torch_gen = torch.Generator().manual_seed(args.encoder_seed + fold * 1000 + epoch)
        loader = DataLoader(epoch_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, generator=torch_gen,
                            pin_memory=(device.type == "cuda"))
        model.train()
        total, batches = 0.0, 0
        for x_d, x_s, y, q_stds in loader:
            x_d, x_s = x_d.to(device), x_s.to(device)
            y, q_stds = y.to(device), q_stds.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_d, x_s)
            loss = loss_fn(pred, y, q_stds)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite encoder loss fold={fold} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        if not batches:
            raise RuntimeError("No encoder batches were produced")
        avg = total / batches
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), best)
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": avg, "best_train_loss": best_loss}) + "\n")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(), "best_loss": best_loss,
                    "config_hash": signature(config)}, latest)
        print(f"[encoder] fold={fold} epoch={epoch}/{args.epochs} loss={avg:.6f} best={best_loss:.6f}", flush=True)
    if not best.exists():
        raise RuntimeError("No best encoder weights produced")
    write_json(out / "TRAIN_DONE.json", {"finished": True, "epochs": args.epochs,
                                           "best_training_loss": best_loss})
    return best, norm_file


def feature_dates(seq_length: int, n: int) -> pd.DatetimeIndex:
    # Entire input window stays within the supervised training period.
    all_dates = pd.date_range(pd.Timestamp(TRAIN_START) + pd.Timedelta(days=seq_length - 1),
                              pd.Timestamp(TRAIN_END), freq="D")
    if len(all_dates) < 1:
        raise RuntimeError("Insufficient training days for the chosen sequence length")
    if n == 0 or n >= len(all_dates):
        return all_dates
    ids = np.unique(np.linspace(0, len(all_dates) - 1, n, dtype=int))
    return all_dates[ids]


def extract_and_cluster(args: argparse.Namespace, fold: int, train: list[str], test: list[str],
                        full_db: Path, scaler: dict, device: torch.device) -> None:
    out = args.output_root / f"fold_{fold}"
    encoder_dir = out / "encoder"
    best = encoder_dir / "model_best.pt"
    norm_cfg = read_json(encoder_dir / "static_norm.json")
    columns = list(norm_cfg["columns"])
    all_ids = train + test
    attrs = load_all_statics(full_db, all_ids, columns)
    means = pd.Series(norm_cfg["means"]).reindex(columns).astype(float)
    stds = pd.Series(norm_cfg["stds"]).reindex(columns).astype(float)
    if (stds <= 0).any():
        raise RuntimeError("Invalid saved static scaler")
    attrs_norm = (attrs - means) / stds
    model = create_model(args, len(columns), device)
    try:
        state = torch.load(best, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(best, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    dates = feature_dates(args.seq_length, args.embedding_samples)
    dates_all = pd.date_range(dates.min() - pd.Timedelta(days=args.seq_length - 1), dates.max(), freq="D")
    ends = ((dates - dates_all[0]).days).to_numpy()
    x_mean = np.asarray(scaler["input_means"], dtype=np.float32)
    x_std = np.asarray(scaler["input_stds"], dtype=np.float32)
    if x_std.shape != (5,) or np.any(x_std <= 0):
        raise RuntimeError("Invalid saved dynamic scaler")
    rows = []
    with torch.inference_mode():
        for idx, basin in enumerate(all_ids, 1):
            forcing, _area = load_forcing(args.camels_root, basin)
            window = forcing.reindex(dates_all)[DYNAMIC].to_numpy(dtype=np.float32)
            if not np.isfinite(window).all():
                raise RuntimeError(f"Incomplete training-period meteorology in {basin}")
            xn = (window - x_mean) / x_std
            basin_stat = attrs_norm.loc[basin].to_numpy(dtype=np.float32)
            embedding_batches = []
            for start in range(0, len(ends), args.infer_batch_size):
                cut = ends[start:start + args.infer_batch_size]
                x = np.stack([xn[j - args.seq_length + 1:j + 1, :] for j in cut])
                s = np.repeat(basin_stat[None, :], len(cut), axis=0)
                z = model.encode_only(torch.from_numpy(x).to(device), torch.from_numpy(s).to(device))
                embedding_batches.append(z.cpu().numpy())
            emb = np.concatenate(embedding_batches).mean(axis=0)
            if not np.isfinite(emb).all():
                raise RuntimeError(f"Invalid embedding for {basin}")
            rows.append({"basin_id": basin, **{f"emb_{j+1:02d}": float(v) for j, v in enumerate(emb)}})
            if idx == 1 or idx % 50 == 0 or idx == len(all_ids):
                print(f"[embedding] fold={fold} {idx}/{len(all_ids)}", flush=True)

    emb_df = pd.DataFrame(rows).set_index("basin_id")
    emb_cols = list(emb_df.columns)
    if set(emb_df.index) != set(all_ids) or len(emb_cols) != args.encoding_dim:
        raise RuntimeError("Incomplete basin embeddings")
    standardizer = StandardScaler().fit(emb_df.loc[train, emb_cols].values)
    train_z = standardizer.transform(emb_df.loc[train, emb_cols].values)
    test_z = standardizer.transform(emb_df.loc[test, emb_cols].values)
    kmeans = KMeans(n_clusters=args.k, random_state=args.kmeans_seed, n_init=args.n_init)
    train_labels = kmeans.fit_predict(train_z)
    test_labels = kmeans.predict(test_z)
    counts = np.bincount(train_labels, minlength=args.k)
    if np.any(counts < args.min_cluster_size):
        raise RuntimeError(f"Too-small training clusters: {counts.tolist()}; choose other configuration")
    config = {"algorithm": "supervised_encoder_kmeans", "train_basins": train, "test_basins": test,
              "encoder_config_hash": signature(read_json(encoder_dir / "run_cfg.json")),
              "embedding_samples": int(len(dates)), "embedding_dates_first": str(dates[0].date()),
              "embedding_dates_last": str(dates[-1].date()), "embedding_sampling": "evenly_spaced_training_days",
              "k": args.k, "kmeans_seed": args.kmeans_seed, "n_init": args.n_init,
              "min_cluster_size": args.min_cluster_size, "fit_scope": "train_basins_only",
              "heldout_input": "static_and_meteorology_only"}
    cfg_path = out / "clustering_manifest.json"
    if cfg_path.exists() and read_json(cfg_path) != config:
        raise RuntimeError(f"Existing clustering configuration differs: {cfg_path}; use a NEW output_root")
    for filename in ["basin_cluster_table_gauged.csv", "ungauged_pred_clusters.csv", "basin_embeddings.csv"]:
        if (out / filename).exists() and not cfg_path.exists():
            raise RuntimeError(f"Orphan grouping file exists: {out / filename}; refusing overwrite")
    write_json(cfg_path, config)
    emb_df.reset_index().to_csv(out / "basin_embeddings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"basin_id": train, "cluster_id": train_labels}).to_csv(
        out / "basin_cluster_table_gauged.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"basin_id": test, "pred_cluster": test_labels}).to_csv(
        out / "ungauged_pred_clusters.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out / "cluster_params.npz", feature_means=standardizer.mean_,
                        feature_scales=standardizer.scale_, centers=kmeans.cluster_centers_)
    write_json(out / "cluster_summary.json", {
        "n_gauged": len(train), "n_ungauged": len(test), "k": args.k,
        "train_cluster_sizes": counts.tolist(),
        "ungauged_cluster_sizes": np.bincount(test_labels, minlength=args.k).tolist(),
    })
    print(f"[cluster] fold={fold} K={args.k} train cluster sizes={counts.tolist()}")
    print(f"[output] {out / 'basin_cluster_table_gauged.csv'}")
    print(f"[output] {out / 'ungauged_pred_clusters.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["train", "cluster", "all"], default="all")
    p.add_argument("--camels_root", type=Path, default=ROOT / "data" / "camels_root")
    p.add_argument("--split_root", type=Path, default=ROOT / "data" / "splits")
    p.add_argument("--output_root", type=Path, default=ROOT / "outputs" / "encoder_clustering")
    p.add_argument("--folds", nargs="+", type=int, default=[1])
    p.add_argument("--seq_length", type=int, default=270)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--encoding_dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--forget_bias", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--samples_per_epoch", type=int, default=0, help="0 means full training samples; >0 samples with replacement")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--encoder_seed", type=int, default=12345)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--kmeans_seed", type=int, default=12345)
    p.add_argument("--n_init", type=int, default=20)
    p.add_argument("--min_cluster_size", type=int, default=1)
    p.add_argument("--embedding_samples", type=int, default=50, help="0 means use all training-period days")
    p.add_argument("--infer_batch_size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    for name in ["seq_length", "hidden_size", "encoding_dim", "epochs", "batch_size", "k", "n_init", "infer_batch_size"]:
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.samples_per_epoch < 0 or args.embedding_samples < 0 or args.num_workers < 0:
        p.error("Negative sample counts or workers are invalid")
    if not (0 <= args.dropout < 1) or args.lr <= 0 or args.grad_clip <= 0:
        p.error("Invalid dropout, lr or grad_clip")
    if not args.folds or len(args.folds) != len(set(args.folds)) or min(args.folds) < 1:
        p.error("--folds must contain unique positive IDs")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.camels_root = args.camels_root.resolve()
    args.split_root = args.split_root.resolve()
    args.output_root = args.output_root.resolve()
    if not args.camels_root.is_dir():
        raise FileNotFoundError(args.camels_root)
    if not args.split_root.is_dir():
        raise FileNotFoundError(args.split_root)
    # Reuse exactly the fold-only preprocessing from the released Global script.
    global_helpers.SEQ_LENGTH = args.seq_length
    for fold in args.folds:
        train, test = get_split(args.split_root, fold)
        if len(train) < args.k and args.stage != "train":
            raise RuntimeError(f"fold={fold}: K={args.k} > number of training basins={len(train)}")
        print(f"[fold {fold}] gauged={len(train)} heldout={len(test)} device={device}", flush=True)
        if args.stage in ("train", "all"):
            filtered_db, h5_path, scaler_path, scaler = global_helpers.prepare_shared_data(
                args.camels_root, args.output_root, fold, train, False)
            train_encoder(args, fold, train, filtered_db, h5_path, scaler, device)
        if args.stage in ("cluster", "all"):
            shared = args.output_root / "shared_data" / f"fold_{fold}"
            raw_db = shared / "attributes.db"
            scaler_path = shared / "dynamic_output_scaler.json"
            if not raw_db.exists() or not scaler_path.exists():
                raise FileNotFoundError(f"Missing encoder preparation data under {shared}; run --stage train first")
            enc_cfg = read_json(args.output_root / f"fold_{fold}" / "encoder" / "run_cfg.json")
            for key in ("hidden_size", "encoding_dim", "seq_length", "dropout", "forget_bias"):
                if enc_cfg[key] != getattr(args, key):
                    raise RuntimeError(f"--{key} disagrees with trained encoder; supply the matching settings")
            scaler = read_json(scaler_path)
            extract_and_cluster(args, fold, train, test, raw_db, scaler, device)


if __name__ == "__main__":
    main()
