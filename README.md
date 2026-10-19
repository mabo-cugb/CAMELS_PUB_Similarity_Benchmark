# CAMELS PUB similarity benchmark — public reproduction scripts

This repository accompanies the study of basin-similarity grouping and transfer strategies for prediction in ungauged basins (PUB) using 531 CAMELS-US basins. The scripts cover **G**, **ST**, **PTST** and **MST**; the names **PTST** and **MST** follow the manuscript. Group assignments are *prepared inputs*, not produced by the training scripts.

## Structure and data

Put the five Python scripts and this README at repository root, with the unchanged original `src/` package (especially `src/datautils.py`, `src/datasets.py`, `src/lstm.py`, `src/nseloss.py`, `src/utils.py`, `src/metrics.py`). The Python scripts import **`src`**, not `papercode`.

```text
repository/
  train_global_lstm.py
  train_cluster_lstm.py
  meta_pretrain_lstm.py
  finetune_cluster_lstm.py
  evaluate_models.py
  src/
  data/
    camels_root/                # original CAMELS-US forcing, flow and attributes
    splits/
      fold_1/split/gauged_basins.txt
      fold_1/split/ungauged_basins.txt
      fold_2/split/...         # only needed for five-fold runs
    clustering/
      seed12345/fold_1/basin_cluster_table_gauged.csv
      seed12345/fold_1/ungauged_pred_clusters.csv
      ...
  outputs/                     # generated during training and evaluation
```

Each `gauged_basins.txt` and `ungauged_basins.txt` contains one 8-digit USGS basin ID per line. For an experiment with a single fixed 513/18 split, create **only** `fold_1/split/` and run every script with `--folds 1`. For the 531-basin fivefold experiment, omit `--folds` to run folds 1–5. Keep the two experiments in **different `--output_root` locations**. To reuse a grouping model for other prepared clusterings, pass `--cluster_root`; use `--n_clusters` to match their label count. No clustering algorithm is executed by this repository.

The cluster table contains `basin_id,cluster_id`, one row for each training basin, with cluster IDs `0` through `K-1`. `ungauged_pred_clusters.csv` contains `basin_id,pred_cluster` for the held-out basins. Cluster table and category mapping paths can be shared by all seeds via `data/clustering/fold_1/...` when the assignments are seed-invariant.

**Default dates:** train 1989-10-01 to 1999-09-30; evaluation 1999-10-01 to 2004-09-30. Inputs are the five CAMELS meteorological forcings plus 27 static attributes; output is daily runoff depth (mm/day). The model has a 270-day lookback, dropout 0.4 and NSE-style loss. These data and time windows match the supplied scripts. The CAMELS data themselves are not redistributed here; obtain them from their original provider.

## Publication defaults

| Model/stage | Hidden size | Maximum epochs | Optimizer/LR | Selection |
| --- | ---: | ---: | --- | --- |
| G | 256 | 30 | Adam; 1e-3 (epochs 1–10), 5e-4 (11–20), 1e-4 (21–30) | final epoch (`model_best.pt` is a legacy filename) |
| ST | size-adaptive, 20–90 | 70 | Adam 1e-3; train-loss plateau, patience 4, factor **0.8**, up to 2 decays | lowest training loss |
| Meta pretraining for MST | 80 | 20 meta epochs | task inner LR 0.005, meta step 0.001 | best within-training task query loss |
| **PTST** fine-tuning | 256 (must match G) | **exactly 10** | **6e-4** | lowest training loss among 10 completed epochs; **no early stopping/rollback** |
| **MST** fine-tuning | 80 (must match Meta) | **at most 70** | 3e-4; existing training-loss rollback schedule | lowest training loss |

**Paper correction:** the original ST script uses `lr_factor=0.8`. The manuscript text giving 0.7 was a typographical error; the public-code default remains **0.8**. MST hyperparameters other than its 80-unit hidden layer and requested maximum 70 fine-tuning epochs have deliberately *not* been redefined; confirm the remaining MST paper settings before declaring an exact reproduction.

Five default seeds: `12345 1234 123 12 1`. ST expert hidden size is adaptive by default (`monotonic_he2024`); use `--hidden_rule strict_he2024` or `--fixed_hidden_size 80` for additional experiments. All scripts support `--help`.

## Run — experiment 2, fivefold 531 basins

Run commands **from the repository root**, in this order (PowerShell one-liners):

```powershell
python -u .\train_global_lstm.py --num_workers 0
python -u .\train_cluster_lstm.py --lr_factor 0.8 --num_workers 0
python -u .\meta_pretrain_lstm.py --hidden_size 80 --num_workers 0
python -u .\finetune_cluster_lstm.py --init_type global --epochs 10 --lr 0.0006 --no-early_stopping --num_workers 0
python -u .\finetune_cluster_lstm.py --init_type meta --epochs 70 --lr 0.0003 --early_stopping --num_workers 0
python -u .\evaluate_models.py --seeds 12345 1234 123 12 1 --save_timeseries
```

For a separate 80-unit MST run when old 256-unit weights exist, use independent directories in **both** commands:

```powershell
python -u .\meta_pretrain_lstm.py --output_root .\outputs\meta_pretrain_h80 --hidden_size 80
python -u .\finetune_cluster_lstm.py --init_type meta --init_root .\outputs\meta_pretrain_h80 --output_root .\outputs\meta_finetune_lstm_h80 --epochs 70
```

Then evaluate with matching flags `--meta_root .\outputs\meta_pretrain_h80 --mst_root .\outputs\meta_finetune_lstm_h80`. Use `--pt_root` to point to an independent PTST run, when needed.

## Run — experiment 1, one fixed 513/18 split

The same training scripts can be reused; only the split and clustering root and output destinations change. For example:

```powershell
python -u .\train_global_lstm.py --folds 1 --split_root .\data\exp1_splits --output_root .\outputs\exp1_global
python -u .\train_cluster_lstm.py --folds 1 --split_root .\data\exp1_splits --cluster_root .\data\exp1_clustering --n_clusters 6 --output_root .\outputs\exp1_cluster
```

These commands are **examples**, not a claim that all seven experiment-1 grouping models can be reproduced merely by changing names: SA/E grouping requires its own prepared assignments/model implementation when structurally different. The evaluation script likewise accepts `--folds 1`, model/cluster directories, and a separate output root. Do not use fivefold outputs as experiment-1 outputs.

## Exposed parameters

* Global: `--folds`, `--seeds`, `--epochs`, `--batch_size`, `--hidden_size`, `--dropout`, `--weight_decay`, `--seq_length`, `--lr_stage1`, `--lr_stage2`, `--lr_stage3`, `--lr_stage1_end`, `--lr_stage2_end`, `--grad_clip`.
* ST: `--folds`, `--seeds`, `--n_clusters`, `--hidden_rule`, `--fixed_hidden_size`, `--epochs`, `--lr`, `--batch_size`, `--dropout`, `--seq_length`, `--lr_patience`, `--lr_factor`, `--min_lr`, `--max_lr_decays`, `--grad_clip`.
* Meta: `--folds`, `--seeds`, `--meta_epochs`, `--meta_batch_size`, `--support_frac`, `--inner_steps`, `--inner_lr`, `--meta_lr`, `--hidden_size`, `--batch_size`, `--dropout`, `--seq_length`, `--grad_clip`.
* Fine-tune: `--init_type global|meta`, `--folds`, `--n_clusters`, `--epochs`, `--lr`, `--early_stopping`/`--no-early_stopping`, `--global_hidden_size`, `--meta_hidden_size`, `--batch_size`, `--dropout`, `--seq_length`, `--lr_patience`, `--lr_factor`, `--min_lr`, `--max_lr_decays`, `--grad_clip`.
* Evaluation: `--models`, `--seeds`, `--folds`, `--camels_root`, `--attributes_db`, `--test_start`, `--test_end`, `--seq_length`, `--batch_size`, `--clip_negative`, `--save_timeseries`, `--allow_missing` and model roots.

`--seq_length` must agree across every linked stage. The scripts reject H5 caches with mismatched sequence length. Changing optimizer, data partition, or model dimension requires **a new output root** rather than resuming old results. Rebuilding shared training H5/scalers under saved model weights is blocked.

## Evaluation and no-leakage boundary

Every model loads its own training-produced `dynamic_output_scaler.json` and `static_norm.json`; **no scaler is fitted on held-out basins**. The evaluation loads held-out basin static values from the **full** original `attributes.db` in the corresponding model's `shared_data/fold_N/` folder, not the training-only `attributes_filtered.db`. You may provide `--attributes_db` pointing to a single full CAMELS attributes database, but it must contain target basins. In either case, the means/stds remain those saved during training.

Each basin's five seed predictions are averaged by date first and then NSE/KGE/etc. are calculated for the ensemble. `--clip_negative` is **off by default** (preserved legacy behavior); when supplied, negative outputs are clipped **per seed before averaging**, which is not equivalent to clipping only the final ensemble. Apply the same setting to all four models.

Artifacts: model checkpoints and per-run `run_cfg.json` are in `outputs/<model>/seed<seed>/fold_<fold>/...`; per-seed/per-basin and five-seed ensemble metrics and paired tests are in `outputs/evaluation/`. `TRAIN_DONE.json` indicates completed training. A restarted run uses `checkpoint_latest.pt` only when saved model/run settings agree with the supplied settings. The default model selection for G differs from ST/PTST/MST; see the table above.

## Dependencies and provenance

Install supported versions of `numpy`, `pandas`, `torch`, `h5py`, `scipy`, `numba` and `tqdm`, as well as CAMELS source dependencies. Make sure `src/` contains the required implementation modules. Retain the attribution and licenses in third-party-derived helper modules (e.g. Kratzert et al., 2019). The training scripts do not implement the clustering process itself. The spatial SHAP analysis is a separate script and is not edited in this six-file update.
