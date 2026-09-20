from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import src.datautils as datautils_module
from src.datautils import INVALID_ATTR, add_camels_attributes, load_attributes, load_discharge, load_forcing
from src.datasets import CamelsH5
from src.lstm import LSTM
from src.nseloss import NSELoss
from src.utils import create_h5_files

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMELS_ROOT = ROOT / "data" / "camels_root"
DEFAULT_SPLIT_ROOT = ROOT / "data" / "splits"
DEFAULT_CLUSTER_ROOT = ROOT / "data" / "clustering"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "cluster_lstm"

TRAIN_START = "1989-10-01"
TRAIN_END = "1999-09-30"
TEST_START = "1999-10-01"
TEST_END = "2004-09-30"

SEEDS = [12345, 1234, 123]
N_FOLDS = 5
N_CLUSTERS = 8
SEQ_LENGTH = 270
BATCH_SIZE = 256
DROPOUT = 0.4
EPOCHS = 70
LR = 1e-3
LR_PATIENCE = 4
LR_FACTOR = 0.8
MIN_LR = 1e-5
MAX_LR_DECAYS = 2
GRAD_CLIP = 1.0
FORGET_GATE_BIAS = 5
DYN_INPUT_SIZE = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DYNAMIC_COLUMNS = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]
SCALER_PROTOCOL = "fold_training_period"


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
    basins = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            basins.append(normalize_basin_id(line))
    return list(dict.fromkeys(basins))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_fold_dynamic_output_scaler(camels_root: Path, basins: list[str]) -> dict[str, Any]:
    x_count = np.zeros(DYN_INPUT_SIZE, dtype=np.int64)
    x_sum = np.zeros(DYN_INPUT_SIZE, dtype=np.float64)
    x_sumsq = np.zeros(DYN_INPUT_SIZE, dtype=np.float64)
    y_count = 0
    y_sum = 0.0
    y_sumsq = 0.0
    start = pd.Timestamp(TRAIN_START)
    end = pd.Timestamp(TRAIN_END)
    for i, basin in enumerate(basins, 1):
        forcing, area = load_forcing(camels_root, basin)
        forcing = forcing.loc[start:end]
        x = forcing[DYNAMIC_COLUMNS].to_numpy(dtype=np.float64)
        finite_x = np.isfinite(x)
        x_count += finite_x.sum(axis=0).astype(np.int64)
        x_safe = np.where(finite_x, x, 0.0)
        x_sum += x_safe.sum(axis=0)
        x_sumsq += np.square(x_safe).sum(axis=0)
        q = load_discharge(camels_root, basin, area).loc[start:end].to_numpy(dtype=np.float64)
        valid_y = np.isfinite(q) & (q >= 0.0)
        q = q[valid_y]
        y_count += int(q.size)
        y_sum += float(q.sum())
        y_sumsq += float(np.square(q).sum())
        if i == 1 or i % 25 == 0 or i == len(basins):
            print(f"[scaler] {i}/{len(basins)}")
    if np.any(x_count == 0) or y_count == 0:
        raise RuntimeError("Unable to compute fold scaler")
    input_means = x_sum / x_count
    input_vars = x_sumsq / x_count - np.square(input_means)
    input_stds = np.sqrt(np.maximum(input_vars, 0.0))
    output_mean = y_sum / y_count
    output_var = y_sumsq / y_count - output_mean ** 2
    output_std = float(np.sqrt(max(output_var, 0.0)))
    if np.any(~np.isfinite(input_means)) or np.any(~np.isfinite(input_stds)) or np.any(input_stds <= 0.0):
        raise RuntimeError("Invalid dynamic-input scaler")
    if not np.isfinite(output_mean) or not np.isfinite(output_std) or output_std <= 0.0:
        raise RuntimeError("Invalid output scaler")
    return {
        "protocol": SCALER_PROTOCOL,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "dynamic_columns": DYNAMIC_COLUMNS,
        "input_means": [float(x) for x in input_means],
        "input_stds": [float(x) for x in input_stds],
        "output_mean": float(output_mean),
        "output_std": float(output_std),
        "input_counts": [int(x) for x in x_count],
        "output_count": int(y_count),
        "n_train_basins": len(basins),
        "train_basins": basins,
    }


def install_scaler(scaler: dict[str, Any]) -> None:
    datautils_module.SCALER = {
        "input_means": np.asarray(scaler["input_means"], dtype=np.float64),
        "input_stds": np.asarray(scaler["input_stds"], dtype=np.float64),
        "output_mean": np.asarray([scaler["output_mean"]], dtype=np.float64),
        "output_std": np.asarray([scaler["output_std"]], dtype=np.float64),
    }


def validate_scaler(scaler: dict[str, Any], gauged_basins: list[str]) -> None:
    if scaler.get("protocol") != SCALER_PROTOCOL:
        raise RuntimeError("Existing scaler protocol does not match current setting")
    if scaler.get("train_start") != TRAIN_START or scaler.get("train_end") != TRAIN_END:
        raise RuntimeError("Existing scaler period does not match current setting")
    if scaler.get("dynamic_columns") != DYNAMIC_COLUMNS:
        raise RuntimeError("Existing scaler dynamic columns do not match current setting")
    saved_basins = list(scaler.get("train_basins", []))
    if saved_basins and set(saved_basins) != set(gauged_basins):
        raise RuntimeError("Existing scaler basin set does not match current fold gauged basins")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def adaptive_hidden_size(n_train_basins: int, rule: str) -> int:
    n = int(n_train_basins)
    if n <= 0:
        raise ValueError("n_train_basins must be positive")
    if rule not in {"monotonic_he2024", "strict_he2024"}:
        raise ValueError(f"Unknown hidden rule: {rule}")
    if n <= 9:
        return 20
    if n <= 14:
        return 30
    if n <= 24:
        return 50
    if n <= 34:
        return 60
    if n <= 44:
        return 70
    if n <= 54:
        return 80
    if n <= 64:
        return 90
    return 80 if rule == "strict_he2024" else 90


def significant_improvement(best: float, current: float) -> bool:
    if best > 0.1:
        threshold = max(0.001, best * 0.002)
    elif best > 0.05:
        threshold = max(0.0005, best * 0.003)
    elif best > 0.02:
        threshold = max(0.0003, best * 0.004)
    else:
        threshold = max(0.0002, best * 0.005)
    return (best - current) > threshold


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def find_split_dir(split_root: Path, fold: int) -> Path:
    candidates = [
        split_root / f"fold_{fold}" / "split",
        split_root / f"fold_{fold}",
    ]
    for path in candidates:
        if (path / "gauged_basins.txt").exists() and (path / "ungauged_basins.txt").exists():
            return path
    raise FileNotFoundError(f"Missing split files for fold_{fold} under {split_root}")


def load_splits(split_root: Path) -> dict[int, tuple[list[str], list[str]]]:
    folds = {}
    reference_all = None
    seen_test = []
    for fold in range(1, N_FOLDS + 1):
        split_dir = find_split_dir(split_root, fold)
        gauged = read_basin_list(split_dir / "gauged_basins.txt")
        ungauged = read_basin_list(split_dir / "ungauged_basins.txt")
        if set(gauged) & set(ungauged):
            raise RuntimeError(f"fold_{fold}: gauged and ungauged overlap")
        fold_all = sorted(set(gauged) | set(ungauged))
        if reference_all is None:
            reference_all = fold_all
        elif fold_all != reference_all:
            raise RuntimeError(f"fold_{fold}: basin set differs from fold_1")
        folds[fold] = (gauged, ungauged)
        seen_test.extend(ungauged)
    if reference_all is None:
        raise RuntimeError("No folds found")
    if len(seen_test) != len(set(seen_test)):
        raise RuntimeError("A test basin appears in more than one fold")
    if sorted(seen_test) != reference_all:
        raise RuntimeError("Five-fold test basins do not cover the full basin set exactly once")
    return folds


def find_cluster_table(cluster_root: Path, seed: int, fold: int) -> Path:
    candidates = [
        cluster_root / f"seed{seed}" / f"fold_{fold}" / "basin_cluster_table_gauged.csv",
        cluster_root / f"seed{seed}" / f"fold_{fold}" / "clustering" / "basin_cluster_table_gauged.csv",
        cluster_root / f"fold_{fold}" / "basin_cluster_table_gauged.csv",
        cluster_root / f"fold_{fold}" / "clustering" / "basin_cluster_table_gauged.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing basin_cluster_table_gauged.csv for seed={seed}, fold={fold} under {cluster_root}"
    )


def load_cluster_members(cluster_root: Path, seed: int, fold: int, gauged: list[str]) -> dict[int, list[str]]:
    path = find_cluster_table(cluster_root, seed, fold)
    df = pd.read_csv(path, dtype={"basin_id": str})
    required = {"basin_id", "cluster_id"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{path} must contain basin_id and cluster_id")
    df = df[["basin_id", "cluster_id"]].copy()
    df["basin_id"] = df["basin_id"].map(normalize_basin_id)
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    if df["basin_id"].duplicated().any():
        raise RuntimeError(f"Duplicate basin ids in {path}")
    if not set(df["cluster_id"]).issubset(set(range(N_CLUSTERS))):
        raise RuntimeError(f"Invalid cluster ids in {path}")
    if set(df["basin_id"]) != set(gauged):
        missing = sorted(set(gauged) - set(df["basin_id"]))
        extra = sorted(set(df["basin_id"]) - set(gauged))
        raise RuntimeError(f"Cluster table does not match gauged basins. missing={missing[:5]}, extra={extra[:5]}")
    members = {}
    for cid in range(N_CLUSTERS):
        basins = sorted(df.loc[df["cluster_id"] == cid, "basin_id"].tolist())
        if not basins:
            raise RuntimeError(f"seed={seed} fold={fold} cluster={cid} has no gauged basins")
        members[cid] = basins
    print(f"[cluster] seed={seed} fold={fold} source={path}")
    print("[cluster] " + " ".join(f"c{cid}={len(members[cid])}" for cid in range(N_CLUSTERS)))
    return members


def prepare_fold_data(camels_root: Path, output_root: Path, fold: int, gauged: list[str], force: bool) -> tuple[Path, Path, Path, dict[str, Any]]:
    data_dir = output_root / "shared_data" / f"fold_{fold}"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_db = data_dir / "attributes.db"
    filtered_db = data_dir / "attributes_filtered.db"
    scaler_path = data_dir / "dynamic_output_scaler.json"
    h5_path = data_dir / "train.h5"
    if force:
        for path in [raw_db, filtered_db, scaler_path, h5_path]:
            if path.exists():
                path.unlink()
    if not raw_db.exists():
        add_camels_attributes(camels_root, db_path=str(raw_db))
    if not filtered_db.exists():
        attrs = load_attributes(str(raw_db), gauged, drop_lat_lon=False)
        attrs.index = [normalize_basin_id(x) for x in attrs.index]
        attrs = attrs.drop(columns=[c for c in INVALID_ATTR if c in attrs.columns], errors="ignore")
        attrs = attrs.select_dtypes(include=[np.number])
        cols = attrs.columns.tolist()
        latlon = [c for c in ["gauge_lat", "gauge_lon"] if c in cols]
        rest = [c for c in cols if c not in latlon]
        attrs = attrs[latlon + rest]
        with sqlite3.connect(str(filtered_db)) as conn:
            attrs.to_sql("basin_attributes", conn, if_exists="replace", index_label="gauge_id")
    if scaler_path.exists():
        scaler = load_json(scaler_path)
        validate_scaler(scaler, gauged)
    else:
        scaler = compute_fold_dynamic_output_scaler(camels_root, gauged)
        save_json(scaler_path, scaler)
    install_scaler(scaler)
    if not h5_path.exists():
        create_h5_files(
            camels_root=camels_root,
            out_file=h5_path,
            basins=gauged,
            dates=[pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)],
            with_basin_str=True,
            seq_length=SEQ_LENGTH,
        )
    return filtered_db, h5_path, scaler_path, scaler


class ClusterLSTMModel(nn.Module):
    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size_dyn + input_size_stat,
            hidden_size=hidden_size,
            batch_first=True,
            initial_forget_bias=FORGET_GATE_BIAS,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        if x_s.dim() == 3:
            x_s = x_s.squeeze(1)
        batch, steps, _ = x_d.shape
        x_s_rep = x_s.unsqueeze(1).expand(batch, steps, x_s.shape[-1])
        x = torch.cat([x_d, x_s_rep], dim=-1)
        h_seq, _ = self.lstm(x)
        return self.fc(self.dropout(h_seq[:, -1, :]))


def validate_existing_cfg(out_dir: Path, seed: int, fold: int, cid: int, hidden_size: int, hidden_rule: str) -> None:
    cfg_path = out_dir / "run_cfg.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    expected = {
        "seed": seed,
        "fold": fold,
        "cluster_id": cid,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "seq_length": SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "hidden_size": hidden_size,
        "hidden_rule": hidden_rule,
        "dropout": DROPOUT,
        "lr": LR,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "min_lr": MIN_LR,
        "max_lr_decays": MAX_LR_DECAYS,
        "epochs": EPOCHS,
        "grad_clip": GRAD_CLIP,
    }
    mismatches = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Existing run_cfg.json conflicts with current settings: {mismatches}")


def train_one_cluster(
    seed: int,
    fold: int,
    cid: int,
    member_basins: list[str],
    fold_gauged: list[str],
    h5_path: Path,
    filtered_db: Path,
    scaler_path: Path,
    scaler: dict[str, Any],
    output_root: Path,
    h5_basin_strings: np.ndarray,
    hidden_rule: str,
    num_workers: int,
) -> None:
    out_dir = output_root / f"seed{seed}" / f"fold_{fold}" / "cluster_models" / f"cluster_{cid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = adaptive_hidden_size(len(member_basins), hidden_rule)
    validate_existing_cfg(out_dir, seed, fold, cid, hidden_size, hidden_rule)
    done_path = out_dir / "TRAIN_DONE.json"
    ckpt_path = out_dir / "checkpoint_latest.pt"
    best_path = out_dir / "model_best.pt"
    best_full_path = out_dir / "checkpoint_best.pt"
    log_path = out_dir / "train_log.jsonl"
    if done_path.exists():
        print(f"[skip] seed={seed} fold={fold} cluster={cid}")
        return
    set_seed(seed)
    base_ds = CamelsH5(
        h5_file=h5_path,
        basins=fold_gauged,
        db_path=str(filtered_db),
        concat_static=False,
        cache=False,
        no_static=False,
    )
    df_all = load_attributes(str(filtered_db), fold_gauged, drop_lat_lon=True)
    df_all.index = [normalize_basin_id(x) for x in df_all.index]
    df_cluster = df_all.loc[member_basins]
    means = df_cluster.mean().fillna(0.0)
    stds = df_cluster.std().replace(0.0, 1.0).fillna(1.0)
    base_ds.df = (df_all - means) / stds
    base_ds.attribute_means = means
    base_ds.attribute_stds = stds
    base_ds.attribute_names = base_ds.df.columns
    shutil.copy2(scaler_path, out_dir / "dynamic_output_scaler.json")
    save_json(
        out_dir / "static_norm.json",
        {
            "columns": list(base_ds.df.columns),
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "stds": {k: float(v) for k, v in stds.to_dict().items()},
            "train_basins": member_basins,
        },
    )
    mask = np.isin(h5_basin_strings, np.asarray(member_basins))
    indices = np.nonzero(mask)[0].astype(np.int64)
    if len(indices) == 0:
        raise RuntimeError(f"seed={seed} fold={fold} cluster={cid}: no H5 samples")
    ds = Subset(base_ds, indices)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = ClusterLSTMModel(
        input_size_dyn=DYN_INPUT_SIZE,
        input_size_stat=int(df_all.shape[1]),
        hidden_size=hidden_size,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = NSELoss()
    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    bad_lr = 0
    lr_decays = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        move_optimizer_state(optimizer, DEVICE)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad_lr = int(ckpt.get("bad_lr", 0))
        lr_decays = int(ckpt.get("lr_decays", 0))
        print(f"[resume] seed={seed} fold={fold} cluster={cid} epoch={start_epoch}")
    save_json(
        out_dir / "run_cfg.json",
        {
            "model": "cluster_lstm",
            "seed": seed,
            "fold": fold,
            "cluster_id": cid,
            "n_train_basins": len(member_basins),
            "train_basins": member_basins,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "seq_length": SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "hidden_size": hidden_size,
            "hidden_rule": hidden_rule,
            "dropout": DROPOUT,
            "lr": LR,
            "lr_schedule": "trainloss_plateau_rollback_pat4_factor0.8_max2",
            "lr_patience": LR_PATIENCE,
            "lr_factor": LR_FACTOR,
            "min_lr": MIN_LR,
            "max_lr_decays": MAX_LR_DECAYS,
            "rollback_to_best": True,
            "monitor": "train_loss",
            "epochs": EPOCHS,
            "grad_clip": GRAD_CLIP,
            "early_stopping": True,
            "model_selection": "lowest_train_loss_model_best",
            "validation": "disabled",
            "static_scaler_scope": "fold_cluster",
            "dynamic_output_scaler_scope": SCALER_PROTOCOL,
            "dynamic_output_scaler": str(out_dir / "dynamic_output_scaler.json"),
            "dynamic_input_means": scaler["input_means"],
            "dynamic_input_stds": scaler["input_stds"],
            "output_mean": scaler["output_mean"],
            "output_std": scaler["output_std"],
        },
    )
    stop_reason = "max_epochs"
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            n_batches = 0
            for x_d, x_s, y, q_stds in loader:
                x_d = x_d.to(DEVICE, non_blocking=True)
                x_s = x_s.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                q_stds = q_stds.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_d, x_s)
                loss = loss_fn(pred, y, q_stds)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                loss_sum += float(loss.item())
                n_batches += 1
            train_loss = loss_sum / max(n_batches, 1)
            last_epoch = epoch
            if not np.isfinite(train_loss):
                raise RuntimeError(f"Non-finite train loss: {train_loss}")
            previous_best = float(best_score)
            raw_best = train_loss < best_score
            significant = True if not np.isfinite(best_score) else raw_best and significant_improvement(best_score, train_loss)
            if raw_best:
                best_score = float(train_loss)
                best_epoch = int(epoch)
                torch.save(model.state_dict(), best_path)
                torch.save(
                    {
                        "epoch": int(epoch),
                        "best_score": float(best_score),
                        "best_epoch": int(best_epoch),
                        "lr_decays": int(lr_decays),
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                    },
                    best_full_path,
                )
            bad_lr = 0 if significant else bad_lr + 1
            lr_before = float(optimizer.param_groups[0]["lr"])
            rollback = False
            rollback_best_epoch = None
            early_stop = False
            if bad_lr >= LR_PATIENCE:
                new_lr = max(lr_before * LR_FACTOR, MIN_LR)
                if new_lr < lr_before and lr_decays < MAX_LR_DECAYS:
                    if not best_full_path.exists():
                        raise RuntimeError(f"Missing best checkpoint: {best_full_path}")
                    best_ckpt = torch.load(best_full_path, map_location="cpu")
                    model.load_state_dict(best_ckpt["model_state"])
                    optimizer.load_state_dict(best_ckpt["optim_state"])
                    move_optimizer_state(optimizer, DEVICE)
                    rollback_best_epoch = int(best_ckpt.get("best_epoch", best_ckpt["epoch"]))
                    best_score = min(float(best_score), float(best_ckpt["best_score"]))
                    best_epoch = rollback_best_epoch
                    lr_decays += 1
                    apply_lr(optimizer, new_lr)
                    bad_lr = 0
                    rollback = True
                else:
                    if best_full_path.exists():
                        best_ckpt = torch.load(best_full_path, map_location="cpu")
                        model.load_state_dict(best_ckpt["model_state"])
                        optimizer.load_state_dict(best_ckpt["optim_state"])
                        move_optimizer_state(optimizer, DEVICE)
                        rollback_best_epoch = int(best_ckpt.get("best_epoch", best_ckpt["epoch"]))
                        best_score = min(float(best_score), float(best_ckpt["best_score"]))
                        best_epoch = rollback_best_epoch
                        rollback = True
                    stop_reason = "max_lr_decays" if lr_decays >= MAX_LR_DECAYS else "min_lr_reached"
                    early_stop = True
            rec = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "previous_best_score": float(previous_best),
                "best_score": float(best_score),
                "best_epoch": int(best_epoch),
                "raw_best": bool(raw_best),
                "significant_improvement": bool(significant),
                "bad_lr": int(bad_lr),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "lr_decays": int(lr_decays),
                "rollback": bool(rollback),
                "rollback_best_epoch": rollback_best_epoch,
                "early_stop": bool(early_stop),
            }
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            torch.save(
                {
                    "epoch": int(epoch),
                    "best_score": float(best_score),
                    "best_epoch": int(best_epoch),
                    "bad_lr": int(bad_lr),
                    "lr_decays": int(lr_decays),
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(
                f"[seed {seed} | fold {fold} | c{cid} | epoch {epoch}/{EPOCHS}] "
                f"loss={train_loss:.6f} best={best_score:.6f}@{best_epoch} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} "
                f"bad_lr={bad_lr}/{LR_PATIENCE} decays={lr_decays}/{MAX_LR_DECAYS}"
            )
            if early_stop:
                break
    except KeyboardInterrupt:
        print(f"[interrupt] seed={seed} fold={fold} cluster={cid}")
        raise
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
    save_json(
        done_path,
        {
            "finished": True,
            "reason": stop_reason,
            "last_epoch_ran": int(last_epoch),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "seed": seed,
            "fold": fold,
            "cluster_id": cid,
            "n_train_basins": len(member_basins),
            "hidden_size": hidden_size,
            "dynamic_output_scaler_scope": SCALER_PROTOCOL,
        },
    )
    print(f"[done] seed={seed} fold={fold} cluster={cid} best={best_score:.6f}@{best_epoch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camels_root", type=str, default=str(DEFAULT_CAMELS_ROOT))
    parser.add_argument("--split_root", type=str, default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--cluster_root", type=str, default=str(DEFAULT_CLUSTER_ROOT))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--hidden_rule",
        type=str,
        default="monotonic_he2024",
        choices=["monotonic_he2024", "strict_he2024"],
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_rebuild_fold_data", action="store_true")
    args = parser.parse_args()

    camels_root = Path(args.camels_root)
    split_root = Path(args.split_root)
    cluster_root = Path(args.cluster_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not camels_root.exists():
        raise FileNotFoundError(camels_root)

    folds = load_splits(split_root)
    shared_data = {}
    for fold in range(1, N_FOLDS + 1):
        gauged, _ = folds[fold]
        shared_data[fold] = prepare_fold_data(
            camels_root=camels_root,
            output_root=output_root,
            fold=fold,
            gauged=gauged,
            force=args.force_rebuild_fold_data,
        )

    print(f"[device] {DEVICE}")
    print(f"[split_root] {split_root}")
    print(f"[cluster_root] {cluster_root}")
    print(f"[output_root] {output_root}")

    for seed in args.seeds:
        for fold in range(1, N_FOLDS + 1):
            gauged, ungauged = folds[fold]
            cluster_members = load_cluster_members(cluster_root, seed, fold, gauged)
            filtered_db, h5_path, scaler_path, scaler = shared_data[fold]
            with h5py.File(str(h5_path), "r") as file:
                h5_basin_strings = np.char.decode(file["sample_2_basin"][:], "ascii")
            save_json(
                output_root / f"seed{seed}" / f"fold_{fold}" / "fold_config.json",
                {
                    "seed": seed,
                    "fold": fold,
                    "n_gauged": len(gauged),
                    "n_ungauged": len(ungauged),
                    "cluster_table": str(find_cluster_table(cluster_root, seed, fold)),
                    "train_period": [TRAIN_START, TRAIN_END],
                    "test_period": [TEST_START, TEST_END],
                    "n_clusters": N_CLUSTERS,
                    "dynamic_output_scaler_scope": SCALER_PROTOCOL,
                    "dynamic_output_scaler": str(scaler_path),
                },
            )
            for cid in range(N_CLUSTERS):
                train_one_cluster(
                    seed=seed,
                    fold=fold,
                    cid=cid,
                    member_basins=cluster_members[cid],
                    fold_gauged=gauged,
                    h5_path=h5_path,
                    filtered_db=filtered_db,
                    scaler_path=scaler_path,
                    scaler=scaler,
                    output_root=output_root,
                    h5_basin_strings=h5_basin_strings,
                    hidden_rule=args.hidden_rule,
                    num_workers=args.num_workers,
                )

    print("All cluster LSTM training runs finished.")


if __name__ == "__main__":
    main()
