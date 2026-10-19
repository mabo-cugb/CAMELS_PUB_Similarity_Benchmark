from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import h5py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.datautils as datautils_module
from src.datasets import CamelsH5
from src.datautils import add_camels_attributes, load_attributes, load_discharge, load_forcing
from src.lstm import LSTM
from src.nseloss import NSELoss
from src.utils import create_h5_files

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMELS_ROOT = ROOT / "data" / "camels_root"
DEFAULT_SPLIT_ROOT = ROOT / "data" / "splits"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "global_lstm"

TRAIN_START = "1989-10-01"
TRAIN_END = "1999-09-30"
TEST_START = "1999-10-01"
TEST_END = "2004-09-30"

SEEDS = [12345, 1234, 123,12,1]
N_FOLDS = 5
LR_STAGE1_END = 10
LR_STAGE2_END = 20
SEQ_LENGTH = 270
EPOCHS = 30
BATCH_SIZE = 2000
HIDDEN_SIZE = 256
DROPOUT = 0.4
WEIGHT_DECAY = 0.0
LR_STAGE1 = 1e-3
LR_STAGE2 = 5e-4
LR_STAGE3 = 1e-4
GRAD_CLIP = 1.0
DYN_INPUT_SIZE = 5
FORGET_GATE_BIAS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DYNAMIC_COLUMNS = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]
SCALER_PROTOCOL = "fold_training_period"




def apply_cli_overrides(args: argparse.Namespace, fields: dict[str, str]) -> None:
    """Apply explicit CLI values to the existing training constants without changing helper APIs."""
    for arg_name, constant_name in fields.items():
        globals()[constant_name] = getattr(args, arg_name)


def check_positive_settings() -> None:
    for key in ("EPOCHS", "BATCH_SIZE", "SEQ_LENGTH", "HIDDEN_SIZE"):
        if int(globals()[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not (0 <= DROPOUT < 1):
        raise ValueError("dropout must be in [0, 1)")
    if GRAD_CLIP <= 0:
        raise ValueError("grad_clip must be positive")


def ensure_cache_matches(h5_path: Path, scaler_path: Path, gauged: list[str]) -> None:
    """Do not silently reuse H5 from a different fold/sequence length."""
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            if "input_data" not in f or "sample_2_basin" not in f:
                raise RuntimeError(f"Unrecognized H5 schema: {h5_path}")
            actual_seq = int(f["input_data"].shape[1])
            if actual_seq != SEQ_LENGTH:
                raise RuntimeError(f"{h5_path}: cached seq_length={actual_seq}, requested={SEQ_LENGTH}; use a NEW output_root")
            cached_basins = set(np.char.decode(f["sample_2_basin"][:], "ascii").tolist())
            if not cached_basins.issubset(set(gauged)) or not set(gauged).issubset(cached_basins):
                raise RuntimeError(f"{h5_path}: cached basin set differs from the selected fold; use a NEW output_root")


def protect_shared_rebuild(output_root: Path, fold: int) -> None:
    """Rebuilding inputs underneath trained checkpoints would invalidate resume."""
    found = []
    for p in output_root.glob(f"seed*/fold_{fold}/**/*.pt"):
        found.append(p)
        break
    if found:
        raise RuntimeError(f"Refusing shared-data rebuild for fold_{fold}: model checkpoints exist under {output_root}. Use a NEW output_root")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_basin_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(dict.fromkeys(values))


def find_gauged_list(split_dir: Path) -> Path:
    candidates = [
        "gauged_basins.txt",
        "gauged.txt",
        "train_basins.txt",
        "train.txt",
        "gauged_ids.txt",
        "train_ids.txt",
    ]
    for name in candidates:
        path = split_dir / name
        if path.exists():
            return path
    for path in sorted(split_dir.glob("*.txt")):
        name = path.name.lower()
        if "gauged" in name or "train" in name:
            return path
    raise FileNotFoundError(f"No gauged basin list found in {split_dir}")


def kratzert_lr(epoch: int) -> float:
    if epoch <= LR_STAGE1_END:
        return LR_STAGE1
    if epoch <= LR_STAGE2_END:
        return LR_STAGE2
    return LR_STAGE3


def lr_schedule_label() -> str:
    if (LR_STAGE1, LR_STAGE2, LR_STAGE3, LR_STAGE1_END, LR_STAGE2_END) == (1e-3, 5e-4, 1e-4, 10, 20):
        return "kratzert2019_1e-3_5e-4_1e-4"
    return f"three_stage_{LR_STAGE1:g}_{LR_STAGE2:g}_{LR_STAGE3:g}_at_{LR_STAGE1_END}_{LR_STAGE2_END}"


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def create_filtered_attributes_db(original_db: Path, basins: list[str], filtered_db: Path) -> None:
    attrs = load_attributes(str(original_db), basins, drop_lat_lon=False)
    attrs = attrs.select_dtypes(include=[np.number])
    cols = attrs.columns.tolist()
    latlon = [c for c in ["gauge_lat", "gauge_lon"] if c in cols]
    rest = [c for c in cols if c not in latlon]
    attrs = attrs[latlon + rest]
    filtered_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(filtered_db)) as conn:
        attrs.to_sql("basin_attributes", conn, if_exists="replace", index_label="gauge_id")


def apply_static_norm(ds: CamelsH5, db_path: Path, basins: list[str], out_dir: Path) -> None:
    df = load_attributes(str(db_path), basins, drop_lat_lon=True)
    means = df.mean()
    stds = df.std().replace(0.0, 1.0)
    norm = (df - means) / stds
    ds.df = norm
    ds.attribute_means = means
    ds.attribute_stds = stds
    ds.attribute_names = norm.columns
    save_json(
        out_dir / "static_norm.json",
        {
            "columns": list(norm.columns),
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "stds": {k: float(v) for k, v in stds.to_dict().items()},
            "n_basins": len(basins),
            "scope": "fold_gauged",
        },
    )


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


class GlobalLSTMModel(nn.Module):
    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size_dyn + input_size_stat,
            hidden_size=hidden_size,
            batch_first=True,
            initial_forget_bias=FORGET_GATE_BIAS,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        if x_s.dim() == 3:
            x_s = x_s.squeeze(1)
        seq_len = x_d.shape[1]
        x_s = x_s.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x_d, x_s], dim=-1)
        h, _ = self.lstm(x)
        return self.fc(self.dropout(h[:, -1, :]))


def validate_splits(split_root: Path, selected_folds: list[int] | None = None) -> dict[int, list[str]]:
    if selected_folds is None:
        selected_folds = list(range(1, N_FOLDS + 1))
    if not selected_folds or len(set(selected_folds)) != len(selected_folds):
        raise ValueError("--folds must be a nonempty list of unique positive fold IDs")
    folds: dict[int, list[str]] = {}
    all_test: list[str] = []
    reference_all = None
    for fold in selected_folds:
        if fold < 1:
            raise ValueError("Fold IDs must be positive")
        split_dir = split_root / f"fold_{fold}" / "split"
        if not split_dir.exists():
            split_dir = split_root / f"fold_{fold}"
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        gauged = read_basin_list(find_gauged_list(split_dir))
        ungauged_file = split_dir / "ungauged_basins.txt"
        if not ungauged_file.exists():
            raise FileNotFoundError(ungauged_file)
        ungauged = read_basin_list(ungauged_file)
        if not gauged or not ungauged:
            raise RuntimeError(f"fold_{fold}: empty training or test basin list")
        if set(gauged) & set(ungauged):
            raise RuntimeError(f"fold_{fold}: gauged and ungauged basins overlap")
        union = set(gauged) | set(ungauged)
        if reference_all is None:
            reference_all = union
        elif len(selected_folds) > 1 and union != reference_all:
            raise RuntimeError(f"fold_{fold}: inconsistent basin universe")
        folds[fold] = gauged
        all_test.extend(ungauged)
        print(f"[split] fold={fold} gauged={len(gauged)} ungauged={len(ungauged)}")
    if len(all_test) != len(set(all_test)):
        raise RuntimeError("A test basin appears in more than one fold")
    if selected_folds == list(range(1, N_FOLDS + 1)) and set(all_test) != reference_all:
        raise RuntimeError("Five-fold test lists do not cover the complete basin set")
    print(f"[split] unique test basins={len(set(all_test))}")
    return folds


def prepare_shared_data(
    camels_root: Path,
    output_root: Path,
    fold: int,
    gauged_basins: list[str],
    force_rebuild: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    shared_dir = output_root / "shared_data" / f"fold_{fold}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    raw_db = shared_dir / "attributes.db"
    filtered_db = shared_dir / "attributes_filtered.db"
    scaler_path = shared_dir / "dynamic_output_scaler.json"
    h5_path = shared_dir / "train.h5"
    if force_rebuild:
        protect_shared_rebuild(output_root, fold)
        for path in [raw_db, filtered_db, scaler_path, h5_path]:
            if path.exists():
                path.unlink()
    if not raw_db.exists():
        add_camels_attributes(camels_root, db_path=str(raw_db))
    if not filtered_db.exists():
        create_filtered_attributes_db(raw_db, gauged_basins, filtered_db)
    if scaler_path.exists():
        scaler = load_json(scaler_path)
        validate_scaler(scaler, gauged_basins)
    else:
        scaler = compute_fold_dynamic_output_scaler(camels_root, gauged_basins)
        save_json(scaler_path, scaler)
    install_scaler(scaler)
    if not h5_path.exists():
        create_h5_files(
            camels_root=camels_root,
            out_file=h5_path,
            basins=gauged_basins,
            dates=[pd.to_datetime(TRAIN_START), pd.to_datetime(TRAIN_END)],
            with_basin_str=True,
            seq_length=SEQ_LENGTH,
        )
    ensure_cache_matches(h5_path, scaler_path, gauged_basins)
    return filtered_db, h5_path, scaler_path, scaler


def validate_existing_run(out_dir: Path, seed: int, fold: int) -> None:
    cfg_path = out_dir / "run_cfg.json"
    if not cfg_path.exists():
        return
    cfg = load_json(cfg_path)
    expected = {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "seed": seed,
        "fold": fold,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "lr_schedule": lr_schedule_label(),
        "dynamic_output_scaler_scope": SCALER_PROTOCOL,
    }
    expected.update({"seq_length": SEQ_LENGTH, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                     "lr_epoch_1_10": LR_STAGE1, "lr_epoch_11_20": LR_STAGE2, "lr_epoch_21_30": LR_STAGE3})
    for key, value in expected.items():
        if key in cfg and cfg.get(key) != value:
            raise RuntimeError(f"Existing run_cfg.json does not match current setting: {key}; use a NEW output_root")


def train_one(
    seed: int,
    fold: int,
    gauged_basins: list[str],
    filtered_db: Path,
    h5_path: Path,
    scaler_path: Path,
    scaler: dict[str, Any],
    out_dir: Path,
    num_workers: int,
) -> None:
    validate_existing_run(out_dir, seed, fold)
    if not (out_dir / "run_cfg.json").exists() and any((out_dir / p).exists() for p in ("checkpoint_latest.pt", "model_best.pt", "TRAIN_DONE.json")):
        raise RuntimeError(f"Old checkpoints without run_cfg.json: {out_dir}; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "TRAIN_DONE.json"
    ckpt_path = out_dir / "checkpoint_latest.pt"
    model_path = out_dir / "model_best.pt"
    log_path = out_dir / "train_log.jsonl"
    if done_path.exists():
        print(f"[skip] seed={seed} fold={fold}")
        return
    set_seed(seed)
    ds = CamelsH5(
        h5_file=h5_path,
        basins=gauged_basins,
        db_path=str(filtered_db),
        concat_static=False,
        cache=False,
        no_static=False,
    )
    apply_static_norm(ds, filtered_db, gauged_basins, out_dir)
    shutil.copy2(scaler_path, out_dir / "dynamic_output_scaler.json")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    input_size_stat = int(load_attributes(str(filtered_db), gauged_basins[:1], drop_lat_lon=True).shape[1])
    model = GlobalLSTMModel(DYN_INPUT_SIZE, input_size_stat, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    loss_fn = NSELoss()
    start_epoch = 1
    best_score = float("inf")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        move_optimizer_state(optimizer, DEVICE)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", float("inf")))
        print(f"[resume] seed={seed} fold={fold} next_epoch={start_epoch}")
    save_json(
        out_dir / "run_cfg.json",
        {
            "model": "global_lstm",
            "seed": seed,
            "fold": fold,
            "num_gauged": len(gauged_basins),
            "train_basins": gauged_basins,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "seq_length": SEQ_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "lr_schedule": lr_schedule_label(),
            "lr_epoch_1_10": LR_STAGE1,
            "lr_epoch_11_20": LR_STAGE2,
            "lr_epoch_21_30": LR_STAGE3,
            "lr_stage1_end": LR_STAGE1_END,
            "lr_stage2_end": LR_STAGE2_END,
            "grad_clip": GRAD_CLIP,
            "model_selection": f"final_epoch_{EPOCHS}",
            "h5": relative_string(h5_path),
            "attributes_db": relative_string(filtered_db),
            "dynamic_output_scaler": relative_string(out_dir / "dynamic_output_scaler.json"),
            "dynamic_output_scaler_scope": SCALER_PROTOCOL,
            "dynamic_input_means": scaler["input_means"],
            "dynamic_input_stds": scaler["input_stds"],
            "output_mean": scaler["output_mean"],
            "output_std": scaler["output_std"],
        },
    )
    print(f"[train] seed={seed} fold={fold} basins={len(gauged_basins)} samples={len(ds)}")
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            last_epoch = epoch
            set_lr(optimizer, kratzert_lr(epoch))
            model.train()
            total_loss = 0.0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            train_loss = total_loss / max(1, n_batches)
            if np.isfinite(train_loss):
                best_score = min(best_score, train_loss)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}) + "\n")
            torch.save(
                {
                    "epoch": epoch,
                    "best_score": best_score,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            torch.save(model.state_dict(), model_path)
            print(f"[seed {seed} | fold {fold} | epoch {epoch}/{EPOCHS}] loss={train_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}")
    except KeyboardInterrupt:
        print(f"[interrupt] seed={seed} fold={fold}")
        raise
    save_json(
        done_path,
        {
            "finished": True,
            "reason": "max_epochs",
            "epoch": last_epoch,
            "best_score": best_score,
            "model_selection": f"final_epoch_{EPOCHS}",
            "seed": seed,
            "fold": fold,
            "dynamic_output_scaler_scope": SCALER_PROTOCOL,
        },
    )
    print(f"[done] seed={seed} fold={fold}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camels_root", type=Path, default=DEFAULT_CAMELS_ROOT)
    parser.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, N_FOLDS + 1)), help="Select one or more split folds, e.g. --folds 1")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seq_length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--lr_stage1", type=float, default=LR_STAGE1)
    parser.add_argument("--lr_stage2", type=float, default=LR_STAGE2)
    parser.add_argument("--lr_stage3", type=float, default=LR_STAGE3)
    parser.add_argument("--lr_stage1_end", type=int, default=LR_STAGE1_END)
    parser.add_argument("--lr_stage2_end", type=int, default=LR_STAGE2_END)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_rebuild_shared_data", action="store_true")
    args = parser.parse_args()
    apply_cli_overrides(args, {name.lower(): name for name in (
        "EPOCHS", "BATCH_SIZE", "HIDDEN_SIZE", "DROPOUT", "WEIGHT_DECAY", "SEQ_LENGTH", "GRAD_CLIP",
        "LR_STAGE1", "LR_STAGE2", "LR_STAGE3", "LR_STAGE1_END", "LR_STAGE2_END")})
    check_positive_settings()
    if not 0 < LR_STAGE1_END < LR_STAGE2_END or min(LR_STAGE1, LR_STAGE2, LR_STAGE3) <= 0:
        raise ValueError("Learning-rate stages must have increasing end epochs and positive rates")
    camels_root = args.camels_root if args.camels_root.is_absolute() else ROOT / args.camels_root
    split_root = args.split_root if args.split_root.is_absolute() else ROOT / args.split_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if not camels_root.exists():
        raise FileNotFoundError(camels_root)
    if not split_root.exists():
        raise FileNotFoundError(split_root)
    print(f"[device] {DEVICE}")
    print(f"[camels_root] {relative_string(camels_root)}")
    print(f"[split_root] {relative_string(split_root)}")
    print(f"[output_root] {relative_string(output_root)}")
    print(f"[train] {TRAIN_START} -> {TRAIN_END}")
    print(f"[test] {TEST_START} -> {TEST_END}")
    print(f"[seeds] {args.seeds}")
    fold_gauged = validate_splits(split_root, args.folds)
    shared: dict[int, tuple[Path, Path, Path, dict[str, Any]]] = {}
    for fold in args.folds:
        shared[fold] = prepare_shared_data(
            camels_root,
            output_root,
            fold,
            fold_gauged[fold],
            args.force_rebuild_shared_data,
        )
    for seed in args.seeds:
        for fold in args.folds:
            filtered_db, h5_path, scaler_path, scaler = shared[fold]
            install_scaler(scaler)
            train_one(
                seed,
                fold,
                fold_gauged[fold],
                filtered_db,
                h5_path,
                scaler_path,
                scaler,
                output_root / f"seed{seed}" / f"fold_{fold}",
                args.num_workers,
            )


if __name__ == "__main__":
    main()
