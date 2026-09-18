from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datautils import load_attributes
from src.datasets import CamelsH5
from src.lstm import LSTM
from src.nseloss import NSELoss

ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_ROOT = ROOT / "data" / "splits"
DEFAULT_CLUSTER_ROOT = ROOT / "data" / "clustering"
DEFAULT_GLOBAL_ROOT = ROOT / "outputs" / "global_lstm"
DEFAULT_META_ROOT = ROOT / "outputs" / "meta_pretrain"
DEFAULT_GLOBAL_FT_ROOT = ROOT / "outputs" / "pretrain_finetune_lstm"
DEFAULT_META_FT_ROOT = ROOT / "outputs" / "meta_finetune_lstm"

TRAIN_START = "1989-10-01"
TRAIN_END = "1999-09-30"
TEST_START = "1999-10-01"
TEST_END = "2004-09-30"
SEEDS = [12345, 1234, 123]
N_FOLDS = 5
N_CLUSTERS = 8
SEQ_LENGTH = 270
EPOCHS = 30
LR = 3e-4
BATCH_SIZE = 256
HIDDEN_SIZE = 256
DROPOUT = 0.4
WEIGHT_DECAY = 0.0
LR_PATIENCE = 4
LR_FACTOR = 0.7
MIN_LR = 1e-5
MAX_LR_DECAYS = 2
GRAD_CLIP = 1.0
FORGET_GATE_BIAS = 5
DYN_INPUT_SIZE = 5
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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_split_dir(split_root: Path, fold: int) -> Path:
    for path in [split_root / f"fold_{fold}" / "split", split_root / f"fold_{fold}"]:
        if (path / "gauged_basins.txt").exists() and (path / "ungauged_basins.txt").exists():
            return path
    raise FileNotFoundError(f"Missing split files for fold_{fold} under {split_root}")


def load_splits(split_root: Path) -> dict[int, tuple[list[str], list[str]]]:
    folds = {}
    for fold in range(1, N_FOLDS + 1):
        split_dir = find_split_dir(split_root, fold)
        gauged = read_basin_list(split_dir / "gauged_basins.txt")
        ungauged = read_basin_list(split_dir / "ungauged_basins.txt")
        if set(gauged) & set(ungauged):
            raise RuntimeError(f"fold_{fold}: gauged and ungauged overlap")
        folds[fold] = (gauged, ungauged)
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
    raise FileNotFoundError(f"Missing basin_cluster_table_gauged.csv for seed={seed}, fold={fold}")


def load_cluster_members(cluster_root: Path, seed: int, fold: int, gauged: list[str]) -> dict[int, list[str]]:
    path = find_cluster_table(cluster_root, seed, fold)
    df = pd.read_csv(path, dtype={"basin_id": str})
    if not {"basin_id", "cluster_id"}.issubset(df.columns):
        raise RuntimeError(f"{path} must contain basin_id and cluster_id")
    df = df[["basin_id", "cluster_id"]].copy()
    df["basin_id"] = df["basin_id"].map(normalize_basin_id)
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    if set(df["basin_id"]) != set(gauged):
        raise RuntimeError(f"{path} does not match current gauged basins")
    members = {}
    for cid in range(N_CLUSTERS):
        basins = sorted(df.loc[df["cluster_id"] == cid, "basin_id"].tolist())
        if not basins:
            raise RuntimeError(f"seed={seed} fold={fold} cluster={cid} has no gauged basins")
        members[cid] = basins
    return members


def significant_improvement(best: float, current: float) -> bool:
    if not np.isfinite(best):
        return True
    if best > 0.1:
        threshold = max(0.001, best * 0.002)
    elif best > 0.05:
        threshold = max(0.0005, best * 0.003)
    elif best > 0.02:
        threshold = max(0.0003, best * 0.004)
    else:
        threshold = max(0.0002, best * 0.005)
    return (best - current) > threshold


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def move_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(DEVICE)


class LSTMModel(nn.Module):
    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int, dropout: float) -> None:
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
        steps = x_d.shape[1]
        x_s_rep = x_s.unsqueeze(1).expand(-1, steps, -1)
        x = torch.cat([x_d, x_s_rep], dim=-1)
        h_seq, _ = self.lstm(x)
        return self.fc(self.dropout(h_seq[:, -1, :]))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def resolve_init_artifacts(init_type: str, init_root: Path, seed: int, fold: int) -> tuple[Path, Path, Path, Path]:
    shared_dir = init_root / "_shared_fold_data" / f"fold_{fold}"
    filtered_db = shared_dir / "attributes_filtered.db"
    h5_path = shared_dir / "train.h5"
    fold_dir = init_root / f"seed{seed}" / f"fold_{fold}"
    if init_type == "global":
        model_path = fold_dir / "model_best.pt"
        norm_path = fold_dir / "static_norm.json"
    else:
        model_path = fold_dir / "meta_pretrained_best.pt"
        norm_path = fold_dir / "static_norm.json"
    for path in [filtered_db, h5_path, model_path, norm_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    return model_path, filtered_db, h5_path, norm_path


def load_static_norm(path: Path) -> tuple[pd.Series, pd.Series, list[str]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    columns = list(cfg["columns"])
    means = pd.Series({k: float(v) for k, v in cfg["means"].items()}).reindex(columns)
    stds = pd.Series({k: float(v) for k, v in cfg["stds"].items()}).reindex(columns)
    stds = stds.replace(0.0, 1.0).fillna(1.0)
    means = means.fillna(0.0)
    return means, stds, columns


def train_one_cluster(init_type: str, seed: int, fold: int, cid: int, member_basins: list[str], gauged: list[str], model_path: Path, filtered_db: Path, h5_path: Path, norm_path: Path, output_root: Path, num_workers: int) -> None:
    out_dir = output_root / f"seed{seed}" / f"fold_{fold}" / f"cluster_{cid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "TRAIN_DONE.json"
    if done_path.exists():
        print(f"[skip] {init_type} seed={seed} fold={fold} cluster={cid} completed")
        return
    means, stds, columns = load_static_norm(norm_path)
    ds = CamelsH5(
        h5_file=h5_path,
        basins=gauged,
        db_path=str(filtered_db),
        concat_static=False,
        cache=False,
        no_static=False,
    )
    attrs = load_attributes(str(filtered_db), gauged, drop_lat_lon=True)
    attrs.index = [normalize_basin_id(x) for x in attrs.index]
    attrs = attrs.reindex(columns=columns)
    ds.df = (attrs - means) / stds
    ds.attribute_means = means
    ds.attribute_stds = stds
    ds.attribute_names = columns
    with h5py.File(str(h5_path), "r") as f:
        basin_strings = np.char.decode(f["sample_2_basin"][:], "ascii")
    indices = np.where(np.isin(basin_strings, np.asarray(member_basins)))[0].astype(int)
    if len(indices) == 0:
        raise RuntimeError(f"No samples found for cluster {cid}")
    loader = DataLoader(Subset(ds, indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    model = LSTMModel(DYN_INPUT_SIZE, len(columns), HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = NSELoss()
    latest_path = out_dir / "checkpoint_latest.pt"
    best_full_path = out_dir / "checkpoint_best.pt"
    best_path = out_dir / "model_best.pt"
    log_path = out_dir / "train_log.jsonl"
    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    bad_lr = 0
    lr_decays = 0
    if latest_path.exists():
        try:
            ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        move_optimizer_state(optimizer)
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad_lr = int(ckpt.get("bad_lr", 0))
        lr_decays = int(ckpt.get("lr_decays", 0))
    else:
        model.load_state_dict(load_state(model_path), strict=True)
    save_json(
        out_dir / "run_cfg.json",
        {
            "init_type": init_type,
            "init_model": str(model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path),
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
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "lr_patience": LR_PATIENCE,
            "lr_factor": LR_FACTOR,
            "min_lr": MIN_LR,
            "max_lr_decays": MAX_LR_DECAYS,
            "grad_clip": GRAD_CLIP,
            "static_norm_scope": "all_gauged_current_fold",
            "full_parameter_finetune": True,
        },
    )
    stop_reason = "max_epochs"
    last_epoch = start_epoch - 1
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
            loss = loss_fn(model(x_d, x_s), y, q_stds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)
        if not np.isfinite(train_loss):
            raise RuntimeError(f"Non-finite train loss at seed={seed} fold={fold} cluster={cid} epoch={epoch}")
        last_epoch = epoch
        raw_best = train_loss < best_score
        significant = significant_improvement(best_score, train_loss) if raw_best else False
        if not np.isfinite(best_score):
            significant = True
        if raw_best:
            best_score = float(train_loss)
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                best_full_path,
            )
        bad_lr = 0 if significant else bad_lr + 1
        rollback = False
        early_stop = False
        if bad_lr >= LR_PATIENCE:
            current_lr = float(optimizer.param_groups[0]["lr"])
            new_lr = max(current_lr * LR_FACTOR, MIN_LR)
            if new_lr < current_lr and lr_decays < MAX_LR_DECAYS:
                try:
                    best_ckpt = torch.load(best_full_path, map_location="cpu", weights_only=False)
                except TypeError:
                    best_ckpt = torch.load(best_full_path, map_location="cpu")
                model.load_state_dict(best_ckpt["model_state"], strict=True)
                optimizer.load_state_dict(best_ckpt["optim_state"])
                move_optimizer_state(optimizer)
                lr_decays += 1
                set_lr(optimizer, new_lr)
                bad_lr = 0
                rollback = True
            else:
                if best_full_path.exists():
                    try:
                        best_ckpt = torch.load(best_full_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        best_ckpt = torch.load(best_full_path, map_location="cpu")
                    model.load_state_dict(best_ckpt["model_state"], strict=True)
                    optimizer.load_state_dict(best_ckpt["optim_state"])
                    move_optimizer_state(optimizer)
                stop_reason = "max_lr_decays" if lr_decays >= MAX_LR_DECAYS else "min_lr_reached"
                rollback = True
                early_stop = True
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "best_score": best_score, "best_epoch": best_epoch, "lr": float(optimizer.param_groups[0]["lr"]), "bad_lr": bad_lr, "lr_decays": lr_decays, "rollback": rollback, "early_stop": early_stop}, ensure_ascii=False) + "\n")
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_lr": bad_lr,
                "lr_decays": lr_decays,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "extra": {"early_stop": early_stop, "init_type": init_type},
            },
            latest_path,
        )
        print(f"[{init_type} | seed {seed} | fold {fold} | c{cid} | epoch {epoch}/{EPOCHS}] loss={train_loss:.6f} best={best_score:.6f}@{best_epoch} lr={optimizer.param_groups[0]['lr']:.6g}")
        if early_stop:
            break
    save_json(
        done_path,
        {
            "finished": True,
            "reason": stop_reason,
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "seed": seed,
            "fold": fold,
            "cluster_id": cid,
            "init_type": init_type,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_type", choices=["global", "meta"], required=True)
    parser.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cluster_root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--init_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, N_FOLDS + 1)))
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if args.init_root is None:
        args.init_root = DEFAULT_GLOBAL_ROOT if args.init_type == "global" else DEFAULT_META_ROOT
    if args.output_root is None:
        args.output_root = DEFAULT_GLOBAL_FT_ROOT if args.init_type == "global" else DEFAULT_META_FT_ROOT
    folds = load_splits(args.split_root)
    for seed in args.seeds:
        for fold in args.folds:
            gauged, _ = folds[fold]
            members = load_cluster_members(args.cluster_root, seed, fold, gauged)
            model_path, filtered_db, h5_path, norm_path = resolve_init_artifacts(args.init_type, args.init_root, seed, fold)
            for cid in range(N_CLUSTERS):
                train_one_cluster(args.init_type, seed, fold, cid, members[cid], gauged, model_path, filtered_db, h5_path, norm_path, args.output_root, args.num_workers)


if __name__ == "__main__":
    main()
