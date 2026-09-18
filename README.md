# CAMELS PUB Similarity Benchmark

This repository contains the core code used in the CAMELS prediction-in-ungauged-basins experiments.

## Repository structure

```text
CAMELS_PUB_Similarity_Benchmark/
├── train_global_lstm.py
├── train_cluster_lstm.py
├── meta_pretrain_lstm.py
├── finetune_cluster_lstm.py
├── evaluate_models.py
├── analyze_spatial_shap.py
├── data/
└── src/
```

## Scripts

### `train_global_lstm.py`
Trains the Global LSTM (G) for the PUB experiments.

### `train_cluster_lstm.py`
Reads prepared basin cluster assignments and trains an independent LSTM for each time-series cluster (ST). The script does not perform clustering.

### `meta_pretrain_lstm.py`
Performs meta pretraining using the gauged basins and prepared time-series grouping information in each PUB fold.

### `finetune_cluster_lstm.py`
Performs cluster-specific full-parameter fine-tuning from either a Global LSTM initialization or a meta-pretrained initialization.

- `--init_type global`: PT-ST
- `--init_type meta`: MST

### `evaluate_models.py`
Provides a common evaluation pipeline for G, ST, PT-ST, and MST and exports basin-level performance metrics and model-comparison results.

### `analyze_spatial_shap.py`
Analyzes spatial performance heterogeneity using basin attributes and exports out-of-fold SHAP results for G, ST, PT-ST, and MST.

## Data

`data/` stores the CAMELS data path, PUB basin lists, and prepared clustering assignments required by the training and evaluation scripts.

The clustering procedure itself is not included in this repository. Cluster assignments are treated as prepared experimental inputs.

## Source code

`src/` contains the shared data-loading, model, loss-function, metric, and utility modules used by the root-level scripts.

## Outputs

Training, evaluation, and attribution results are written to `outputs/`.

## Reproducibility

Model settings, experimental periods, hyperparameters, and detailed experimental protocols are described in the accompanying manuscript.

## Code attribution

Several modules under `src/` retain components and license headers from the code accompanying Kratzert et al. (2019), *Benchmarking a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling*. Original copyright and license notices should be retained when redistributing derived source files.
