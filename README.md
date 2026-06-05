# brain2behaviour

A lightweight Python package for connectome-based predictive modelling (CPM) — predicting behaviour from brain connectivity data.

## Overview

`brain2behaviour` handles the full CPM pipeline:

1. **Dataset management** — load brain/behaviour/confound data and generate cross-validation folds
2. **Preprocessing** — gaussianize, residualize confounds, and standardize within each fold (no data leakage)
3. **Feature selection** — edge-wise correlation with behaviour, filtered by p-value threshold
4. **Modelling** — linear CPM with positive/negative network summation

## Installation

```bash
pip install -e .
```

## Quick start

```python
from brain2behaviour.dataset import BrainBehaviorDataset
from brain2behaviour.preprocessing import clean_fold
from brain2behaviour.feature_selection import get_CPM_features

# 1. Create dataset
ds = BrainBehaviorDataset(
    brainData="brain.parquet",
    behaviorData="behaviour.csv",
    confounds="confounds.csv",
    ncv_splits=(10,),
    filepath="experiment.pkl",
)

# 2. Generate CV folds (family-aware or naive)
ds.gen_CV_FoldsNaive(frac=0.2, nsplits=10)

# 3. For each fold: clean data, select features, train model
for fold in ds.cv_folds:
    cleaned = clean_fold(ds, fold, encode_cols=["Site"], area_cols=[], volume_cols=[])
    features = get_CPM_features(cleaned, pthresh=0.01)
```

## Modules

| Module | Description |
|---|---|
| `dataset` | `BrainBehaviorDataset` — container for data, CV folds, features, and hyperparameters |
| `preprocessing` | `clean_fold`, `BrainPipeline`, `BehPipeline` — per-fold confound regression and normalization |
| `feature_selection` | `get_CPM_features` — edge-behaviour correlation with p-value filtering |
| `Linear` | Classic CPM linear model |

## Cross-validation

Three CV strategies are supported:

- **Naive** (`gen_CV_FoldsNaive`) — random shuffle split
- **Family-aware via sklearn** (`gen_CV_FamilyFoldsSKlearn`) — keeps family members in the same fold
- **Family-aware legacy** (`gen_CV_FamilyFoldsImplicit`) — replicates preprint results; use seeds `[0,42,19,123,10,69,33,1,1234,9245]`

## Requirements

- Python >= 3.8
- numpy, pandas, scikit-learn, scipy, pyarrow
