# top of file – imports
from pathlib import Path
import argparse, os, multiprocessing as mp
import pandas as pd
from brain2behaviour.dataset import BrainBehaviorDataset 
from brain2behaviour.preprocessing import clean_fold
from brain2behaviour.feature_selection import get_CPM_features

# ------------------------------------------------------------
def get_fold_features(
    datapath,
    fold,
    ncpus=mp.cpu_count() - 1,
    batch_size=256,
    encode_cols=("Gender", "Acquisition","SSAGA_Income","SSAGA_Educ"),
    bin_encode={"Acquisition": 2},
    area_cols=("Larea", "Rarea"),
    volume_cols=("FS_IntraCranial_Vol", "FS_BrainSeg_Vol"),
    gaussianize=True,
    add_squares=True,
    zscore_cols=True,
):
    datapath = Path(datapath).expanduser().resolve()    # <-- ensure Path
    # out_dir  = datapath.parent / "features_tmp"
    out_dir  = datapath.parent/f"{datapath.stem}_features_tmp" ### build path based on dataset name
    out_dir.mkdir(exist_ok=True)

    data = BrainBehaviorDataset.load(datapath)            # <-- fix class name
    print("Loaded:", datapath)

    cleaned = clean_fold(
        data, fold,
        encode_cols=encode_cols,
        bin_encode=bin_encode,
        area_cols=area_cols,
        volume_cols=volume_cols,
        gaussianize=gaussianize,
        add_squares=add_squares,
        zscore_cols=zscore_cols,
    )

    feats = get_CPM_features(
        cleaned,
        pthresh=0.01,
        ncpus=ncpus,
        batch_size=batch_size)

    for key in feats.keys():
        print(key)
        s=pd.Series(feats[key])
        df=s.to_frame(key)
        os.makedirs(f'{out_dir}/{key}/',exist_ok=True)
        df.to_parquet(f'{out_dir}/{key}/{fold}.parquet')
        print(f'features saved to {out_dir}/{key}/{fold}.parquet')
    
    

# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPM feature selection on one dataset/fold"
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to pickled BrainBehaviorDataset")
    parser.add_argument("--fold", type=str, required=True,
                        help="Fold ID to process (e.g. fold001)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ncpus", type=int,
        default=int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count() - 1)))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    get_fold_features(
        datapath=args.dataset,
        fold=args.fold,
        ncpus=args.ncpus,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()

