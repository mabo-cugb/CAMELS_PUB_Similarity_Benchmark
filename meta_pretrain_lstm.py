from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datautils import INVALID_ATTR, add_camels_attributes, load_attributes
from src.datasets import CamelsH5
from src.lstm import LSTM
from src.nseloss import NSELoss
from src.utils import create_h5_files

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMELS_ROOT = ROOT / "data" / "camels_root"
DEFAULT_SPLIT_ROOT = ROOT / "data" / "splits"
DEFAULT_CLUSTER_ROOT = ROOT / "data" / "clustering"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "meta_pretrain"

TRAIN_START = "1989-10-01"
TRAIN_END = "1999-09-30"
TEST_START = "1999-10-01"
TEST_END = "2004-09-30"
SEEDS = [12345, 1234, 123]
N_FOLDS = 5
N_CLUSTERS = 8
SEQ_LENGTH = 270
META_EPOCHS = 20
META_BATCH_SIZE = 8
SUPPORT_FRAC = 0.8
INNER_STEPS = 5
INNER_LR = 0.005
META_LR = 1e-3
HIDDEN_SIZE = 256
DROPOUT = 0.4
BATCH_SIZE = 256
GRAD_CLIP = 1.0
DYN_INPUT_SIZE = 5
FORGET_GATE_BIAS = 5
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_split_dir(split_root: Path, fold: int) -> Path:
    for path in [split_root / f"fold_{fold}" / "split", split_root / f"fold_{fold}"]:
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
    raise FileNotFoundError(f"Missing basin_cluster_table_gauged.csv for seed={seed}, fold={fold}")


def load_cluster_members(cluster_root: Path, seed: int, fold: int, gauged: list[str]) -> dict[int, list[str]]:
    path = find_cluster_table(cluster_root, seed, fold)
    df = pd.read_csv(path, dtype={"basin_id": str})
    if not {"basin_id", "cluster_id"}.issubset(df.columns):
        raise RuntimeError(f"{path} must contain basin_id and cluster_id")
    df = df[["basin_id", "cluster_id"]].copy()
    df["basin_id"] = df["basin_id"].map(normalize_basin_id)
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    if df["basin_id"].duplicated().any():
        raise RuntimeError(f"Duplicate basin ids in {path}")
    if set(df["basin_id"]) != set(gauged):
        missing = sorted(set(gauged) - set(df["basin_id"]))
        extra = sorted(set(df["basin_id"]) - set(gauged))
        raise RuntimeError(f"Cluster table does not match gauged basins. missing={missing[:5]}, extra={extra[:5]}")
    members = {}
    for cid in sorted(df["cluster_id"].unique().tolist()):
        members[int(cid)] = sorted(df.loc[df["cluster_id"] == cid, "basin_id"].tolist())
    print(f"[cluster] seed={seed} fold={fold} source={path}")
    return members


def create_filtered_attributes_db(raw_db: Path, basins: list[str], filtered_db: Path) -> None:
    attrs = load_attributes(str(raw_db), basins, drop_lat_lon=False)
    attrs.index = [normalize_basin_id(x) for x in attrs.index]
    attrs = attrs.drop(columns=[c for c in INVALID_ATTR if c in attrs.columns], errors="ignore")
    attrs = attrs.select_dtypes(include=[np.number])
    cols = attrs.columns.tolist()
    latlon = [c for c in ["gauge_lat", "gauge_lon"] if c in cols]
    rest = [c for c in cols if c not in latlon]
    attrs = attrs[latlon + rest]
    filtered_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(filtered_db)) as conn:
        attrs.to_sql("basin_attributes", conn, if_exists="replace", index_label="gauge_id")


def prepare_shared_data(camels_root: Path, output_root: Path, fold: int, gauged: list[str], force: bool) -> tuple[Path, Path]:
    shared_dir = output_root / "_shared_fold_data" / f"fold_{fold}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    raw_db = shared_dir / "attributes.db"
    filtered_db = shared_dir / "attributes_filtered.db"
    h5_path = shared_dir / "train.h5"
    if force:
        for path in [raw_db, filtered_db, h5_path]:
            if path.exists():
                path.unlink()
    if not raw_db.exists():
        add_camels_attributes(camels_root, db_path=str(raw_db))
    if not filtered_db.exists():
        create_filtered_attributes_db(raw_db, gauged, filtered_db)
    if not h5_path.exists():
        create_h5_files(
            camels_root=camels_root,
            out_file=h5_path,
            basins=gauged,
            dates=[pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)],
            with_basin_str=True,
            seq_length=SEQ_LENGTH,
        )
    return filtered_db, h5_path


class MetaLSTMModel(nn.Module):
    def __init__(self, input_size_dyn: int, input_size_stat: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
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
        steps = x_d.shape[1]
        x_s_rep = x_s.unsqueeze(1).expand(-1, steps, -1)
        x = torch.cat([x_d, x_s_rep], dim=-1)
        h_seq, _ = self.lstm(x)
        return self.fc(self.dropout(h_seq[:, -1, :]))


def clone_model(model: MetaLSTMModel) -> MetaLSTMModel:
    cloned = MetaLSTMModel(
        input_size_dyn=model.input_size_dyn,
        input_size_stat=model.input_size_stat,
        hidden_size=model.hidden_size,
        dropout=model.dropout_rate,
    ).to(DEVICE)
    cloned.load_state_dict(model.state_dict(), strict=True)
    return cloned


def split_support_query(indices: list[int], support_frac: float, rng: random.Random) -> tuple[list[int], list[int]]:
    idxs = list(indices)
    rng.shuffle(idxs)
    n = len(idxs)
    if n < 2:
        return idxs, idxs
    n_support = int(round(n * support_frac))
    n_support = max(1, min(n - 1, n_support))
    support = idxs[:n_support]
    query = idxs[n_support:]
    return support, query if query else support[:]


def build_basin_order(gauged: list[str], cluster_members: dict[int, list[str]], rng: random.Random) -> list[str]:
    cluster_lists = {}
    assigned = set()
    for cid in sorted(cluster_members):
        cur = [b for b in cluster_members[cid] if b in gauged]
        rng.shuffle(cur)
        if cur:
            cluster_lists[cid] = cur
            assigned.update(cur)
    leftovers = [b for b in gauged if b not in assigned]
    if leftovers:
        rng.shuffle(leftovers)
        cluster_lists[max(cluster_lists.keys(), default=-1) + 1] = leftovers
    ordered = []
    max_len = max((len(v) for v in cluster_lists.values()), default=0)
    cids = sorted(cluster_lists)
    for i in range(max_len):
        round_cids = cids[:]
        rng.shuffle(round_cids)
        for cid in round_cids:
            if i < len(cluster_lists[cid]):
                ordered.append(cluster_lists[cid][i])
    seen = set()
    final = []
    for basin in ordered + gauged:
        if basin not in seen:
            final.append(basin)
            seen.add(basin)
    return final


def run_inner_steps(model: MetaLSTMModel, loader: DataLoader, loss_fn: nn.Module) -> tuple[int, float]:
    loss_sum = 0.0
    steps = 0
    model.train()
    for x_d, x_s, y, q_stds in loader:
        x_d = x_d.to(DEVICE)
        x_s = x_s.to(DEVICE)
        y = y.to(DEVICE)
        q_stds = q_stds.to(DEVICE)
        pred = model(x_d, x_s)
        loss = loss_fn(pred, y, q_stds)
        if not torch.isfinite(loss):
            continue
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= INNER_LR * p.grad
        loss_sum += float(loss.item())
        steps += 1
        if steps >= INNER_STEPS:
            break
    return steps, loss_sum


@torch.no_grad()
def query_loss(model: MetaLSTMModel, loader: DataLoader, loss_fn: nn.Module) -> float:
    total = 0.0
    count = 0
    model.eval()
    for x_d, x_s, y, q_stds in loader:
        x_d = x_d.to(DEVICE)
        x_s = x_s.to(DEVICE)
        y = y.to(DEVICE)
        q_stds = q_stds.to(DEVICE)
        loss = loss_fn(model(x_d, x_s), y, q_stds)
        if torch.isfinite(loss):
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def save_checkpoint(path: Path, model: MetaLSTMModel, epoch: int, best_query_loss: float, history: list[dict[str, Any]], rng: random.Random) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_query_loss": float(best_query_loss),
            "model_state": model.state_dict(),
            "history": history,
            "python_rng_state": rng.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(path: Path, model: MetaLSTMModel, rng: random.Random) -> tuple[int, float, list[dict[str, Any]]]:
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if ckpt.get("python_rng_state") is not None:
        rng.setstate(ckpt["python_rng_state"])
    if ckpt.get("numpy_rng_state") is not None:
        np.random.set_state(ckpt["numpy_rng_state"])
    if ckpt.get("torch_rng_state") is not None:
        torch.random.set_rng_state(ckpt["torch_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in ckpt["cuda_rng_state"]])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_query_loss", float("inf"))), list(ckpt.get("history", []))


def train_one(seed: int, fold: int, gauged: list[str], cluster_members: dict[int, list[str]], filtered_db: Path, h5_path: Path, out_dir: Path, num_workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "TRAIN_DONE.json"
    if done_path.exists():
        print(f"[skip] seed={seed} fold={fold} completed")
        return
    set_seed(seed)
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
    means = attrs.mean().fillna(0.0)
    stds = attrs.std().replace(0.0, 1.0).fillna(1.0)
    ds.df = (attrs - means) / stds
    ds.attribute_means = means
    ds.attribute_stds = stds
    ds.attribute_names = ds.df.columns
    save_json(
        out_dir / "static_norm.json",
        {
            "columns": list(ds.df.columns),
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "stds": {k: float(v) for k, v in stds.to_dict().items()},
            "train_basins": gauged,
        },
    )
    with h5py.File(str(h5_path), "r") as f:
        basin_strings = np.char.decode(f["sample_2_basin"][:], "ascii")
    basin_to_indices = {b: np.where(basin_strings == b)[0].astype(int).tolist() for b in gauged}
    model = MetaLSTMModel(DYN_INPUT_SIZE, len(ds.df.columns), HIDDEN_SIZE, DROPOUT).to(DEVICE)
    loss_fn = NSELoss()
    history_path = out_dir / "meta_pretrain_history.csv"
    latest_path = out_dir / "checkpoint_latest.pt"
    best_ckpt_path = out_dir / "checkpoint_best.pt"
    best_path = out_dir / "meta_pretrained_best.pt"
    final_path = out_dir / "meta_pretrained_final.pt"
    rng = random.Random(seed)
    start_epoch = 1
    best_query = float("inf")
    history = []
    if latest_path.exists():
        last_epoch, best_query, history = load_checkpoint(latest_path, model, rng)
        start_epoch = last_epoch + 1
        print(f"[resume] seed={seed} fold={fold} next_epoch={start_epoch} best_query={best_query:.6f}")
    cfg = {
        "algorithm": "query_loss_weighted_reptile",
        "task_sampling": "cluster_aware",
        "seed": seed,
        "fold": fold,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "seq_length": SEQ_LENGTH,
        "meta_epochs": META_EPOCHS,
        "meta_batch_size": META_BATCH_SIZE,
        "support_frac": SUPPORT_FRAC,
        "inner_steps": INNER_STEPS,
        "inner_lr": INNER_LR,
        "meta_lr": META_LR,
        "meta_lr_schedule": "constant",
        "task_weight": "inverse_sqrt_query_loss",
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "grad_clip": GRAD_CLIP,
        "n_gauged": len(gauged),
    }
    save_json(out_dir / "run_cfg.json", cfg)
    for epoch in range(start_epoch, META_EPOCHS + 1):
        basin_order = build_basin_order(gauged, cluster_members, rng)
        inner_loss_sum = 0.0
        inner_steps_sum = 0
        query_loss_sum = 0.0
        effective_tasks = 0
        outer_updates = 0
        for offset in range(0, len(basin_order), META_BATCH_SIZE):
            task_deltas = []
            task_weights = []
            for basin in basin_order[offset:offset + META_BATCH_SIZE]:
                indices = basin_to_indices.get(basin, [])
                if len(indices) < 2:
                    continue
                support_idx, query_idx = split_support_query(indices, SUPPORT_FRAC, rng)
                support_loader = DataLoader(Subset(ds, support_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
                query_loader = DataLoader(Subset(ds, query_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
                task_model = clone_model(model)
                step_count, task_inner_sum = run_inner_steps(task_model, support_loader, loss_fn)
                if step_count == 0:
                    continue
                q_loss = query_loss(task_model, query_loader, loss_fn)
                if not np.isfinite(q_loss):
                    continue
                delta = {}
                with torch.no_grad():
                    for (name, p_global), (_, p_task) in zip(model.named_parameters(), task_model.named_parameters()):
                        delta[name] = (p_task.data - p_global.data).detach().clone()
                task_deltas.append(delta)
                task_weights.append(1.0 / math.sqrt(max(q_loss, 1e-8)))
                inner_loss_sum += task_inner_sum
                inner_steps_sum += step_count
                query_loss_sum += q_loss
                effective_tasks += 1
            if not task_deltas:
                continue
            weights = np.asarray(task_weights, dtype=np.float64)
            weights /= max(weights.sum(), 1e-12)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    update = torch.zeros_like(p, device=DEVICE)
                    for weight, delta in zip(weights, task_deltas):
                        update += float(weight) * delta[name]
                    p.data += META_LR * update
            outer_updates += 1
        avg_inner = inner_loss_sum / max(inner_steps_sum, 1)
        avg_query = query_loss_sum / max(effective_tasks, 1)
        row = {
            "epoch": epoch,
            "avg_inner_loss": float(avg_inner),
            "avg_query_loss": float(avg_query),
            "effective_tasks": int(effective_tasks),
            "outer_updates": int(outer_updates),
            "basin_coverage": int(len(basin_order)),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
        if avg_query < best_query:
            best_query = float(avg_query)
            torch.save(model.state_dict(), best_path)
            save_checkpoint(best_ckpt_path, model, epoch, best_query, history, rng)
        save_checkpoint(latest_path, model, epoch, best_query, history, rng)
        print(f"[seed {seed} | fold {fold} | meta {epoch}/{META_EPOCHS}] inner={avg_inner:.6f} query={avg_query:.6f} tasks={effective_tasks} updates={outer_updates}")
    torch.save(model.state_dict(), final_path)
    save_json(
        done_path,
        {
            "finished": True,
            "seed": seed,
            "fold": fold,
            "epochs": META_EPOCHS,
            "best_query_loss": best_query,
            "best_model": "meta_pretrained_best.pt",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camels_root", type=Path, default=DEFAULT_CAMELS_ROOT)
    parser.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cluster_root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, N_FOLDS + 1)))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_rebuild_shared_data", action="store_true")
    args = parser.parse_args()
    folds = load_splits(args.split_root)
    shared = {}
    for fold in args.folds:
        gauged, _ = folds[fold]
        shared[fold] = prepare_shared_data(args.camels_root, args.output_root, fold, gauged, args.force_rebuild_shared_data)
    for seed in args.seeds:
        for fold in args.folds:
            gauged, _ = folds[fold]
            cluster_members = load_cluster_members(args.cluster_root, seed, fold, gauged)
            filtered_db, h5_path = shared[fold]
            out_dir = args.output_root / f"seed{seed}" / f"fold_{fold}"
            train_one(seed, fold, gauged, cluster_members, filtered_db, h5_path, out_dir, args.num_workers)


if __name__ == "__main__":
    main()
