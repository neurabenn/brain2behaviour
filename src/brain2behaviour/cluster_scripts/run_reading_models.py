#!/usr/bin/env python3
# top of file – imports
from pathlib import Path
import argparse, os, multiprocessing as mp
import pandas as pd
from brain2behaviour.dataset import BrainBehaviorDataset 
from brain2behaviour.preprocessing import clean_fold
from brain2behaviour.feature_selection import get_CPM_features
from brain2behaviour.Linear.CPM_classic import *
from collections.abc import Sequence

# ------------------------------------------------------------
def run_end2endCPM(datasets, fold, sign, permutation_set, outpath):
    if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
        datasets = list(datasets)
    else:
        datasets = [datasets]
    # Load if path(s), else pass-through
    datasets = [BrainBehaviorDataset.load(str(d)) if isinstance(d, Path) else d
                for d in datasets]

    cleaned_data = prep_ds_CPM_classic(
        datasets=datasets,
        fold=fold,
        sign2keep=sign,
        encode_cols=("Gender", "Acquisition"),
        bin_encode={"Acquisition": 2},
        area_cols=("Larea", "Rarea"),
        volume_cols=("FS_IntraCranial_Vol", "FS_BrainSeg_Vol"),
        gaussianize=True,
        add_squares=True,
        zscore_cols=True
    )

    evaluate_fold_cpm(
        clean_data_dict=cleaned_data,
        outpath=outpath,
        fold=fold,
        perm_set=permutation_set
    )

# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end CPM (prep, train, permute) on one or more datasets."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path, nargs="+", required=True,
        help="Path(s) to pickled BrainBehaviorDataset"
    )
    parser.add_argument(
        "--fold", "-f",
        type=str, required=True,
        help="Fold ID to process (e.g. fold001)"
    )
    parser.add_argument(
        "--sign", "-s",
        choices=["positive", "negative"], required=True,
        help="Which CPM feature sign to keep"
    )
    parser.add_argument(
        "--perms", "-p",
        help="Either an integer number of random shuffles, or a CSV path of index permutations"
    )
    parser.add_argument(
        "--outpath", "-o",
        type=Path, required=True,
        help="Base path (no extension) for saving the model & permutation results"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Interpret --perms as int or path
    try:
        perm_set = int(args.perms)
    except ValueError:
        perm_set = str(args.perms)

    run_end2endCPM(
        datasets=args.dataset,
        fold=args.fold,
        sign=args.sign,
        permutation_set=perm_set,
        outpath=str(args.outpath)
    )


if __name__ == "__main__":
    main()
