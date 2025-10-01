#!/usr/bin/env python3
import sys
import pandas as pd
import os, shutil
from pathlib import Path
from brain2behaviour.dataset import BrainBehaviorDataset
from brain2behaviour.Linear.CPM_classic import prep_ds_CPM_classic,train_predict_test
### gaussianization turned off aug 28th
## tuned back on sept 18th
### turn off after run 


def run_cpm_models(dataset, SES=False):
    r_vals = {s: {} for s in ['joint', 'positive', 'negative']}
    pos_feats = dataset.features['ReadEng_Unadj']['positive']
    neg_feats = dataset.features['ReadEng_Unadj']['negative']

    for fold in dataset.cv_folds.keys():
        if len(pos_feats) == 0 and len(neg_feats) == 0:
            # nothing to run
            r_vals['positive'][fold] = 0
            r_vals['negative'][fold] = 0
            r_vals['joint'][fold] = 0

        elif len(pos_feats) == 0:
            # run only negative
            r_vals['positive'][fold] = 0
            r_vals['joint'][fold] = 0
            clean_cpm = prep_ds_CPM_classic(
                dataset, fold, 'negative',
                encode_cols=["Gender"],
                area_cols=("Larea", "Rarea"),
                volume_cols=("FS_IntraCranial_Vol", "FS_BrainSeg_Vol"),
                bin_encode=None,
                passthrough_cols=None,
                gaussianize=False, add_squares=True, zscore_cols=True
            )
            r, predictions = train_predict_test(clean_cpm)
            r_vals['negative'][fold] = r

        elif len(neg_feats) == 0:
            # run only positive
            r_vals['negative'][fold] = 0
            r_vals['joint'][fold] = 0
            clean_cpm = prep_ds_CPM_classic(
                dataset, fold, 'positive',
                encode_cols=["Gender"],
                area_cols=("Larea", "Rarea"),
                volume_cols=("FS_IntraCranial_Vol", "FS_BrainSeg_Vol"),
                bin_encode=None,
                passthrough_cols=None,
                gaussianize=False, add_squares=True, zscore_cols=True
            )
            r, predictions = train_predict_test(clean_cpm)
            r_vals['positive'][fold] = r

        else:
            # run all three
            for sign in ['joint', 'positive', 'negative']:
                clean_cpm = prep_ds_CPM_classic(
                    dataset, fold, sign,
                    encode_cols=["Gender"],
                    area_cols=("Larea", "Rarea"),
                    volume_cols=("FS_IntraCranial_Vol", "FS_BrainSeg_Vol"),
                    bin_encode=None,
                    passthrough_cols=None,
                    gaussianize=False, add_squares=True, zscore_cols=True
                )
                r, predictions = train_predict_test(clean_cpm)
                r_vals[sign][fold] = r

    return pd.DataFrame(r_vals)


def main(dataset_path: str, feature_dir: str, task: str):
    print(f"[collect] dataset={dataset_path}")
    ds = BrainBehaviorDataset.load(dataset_path)
    print(f"[collect] collecting into {feature_dir}/{task}/")
    ds.collect_features(feature_dir=f"{feature_dir}/{task}/")
    ds.save(overwrite=True)
    print("[collect] done")
    print("predicting joint, positive, and negtive models")
    r_vals=run_cpm_models(ds,SES=True)
    opath_r=str(dataset_path)
    opath_r=opath_r.replace(".pkl","r_vals.csv")
    print(opath_r)
    r_vals.to_csv(opath_r,index=False)
    # cleanup intermediate files
    # --- cleanup: delete al non csv files and folders except  ds_permed0000.pkl which contains actual data and dataset---
    ds_path = Path(dataset_path)
    if ds_path.name != "ds_permed0000.pkl":
        # delete the .pkl file if it exists and is a file
        try:
            if ds_path.exists() and ds_path.is_file():
                ds_path.unlink()
                print(f"[cleanup] removed {ds_path}")
        except Exception as e:
            print(f"[cleanup] could not remove {ds_path}: {e}")

        # delete the *_features_tmp directory if it exists
        tmp_dir = Path(dataset_path.replace(".pkl", "_features_tmp"))
        if tmp_dir.exists() and tmp_dir.is_dir():
            try:
                shutil.rmtree(tmp_dir)
                print(f"[cleanup] removed {tmp_dir}")
            except Exception as e:
                print(f"[cleanup] could not remove {tmp_dir}: {e}")
        

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} DATASET_PATH FEATURE_DIR TASK", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
