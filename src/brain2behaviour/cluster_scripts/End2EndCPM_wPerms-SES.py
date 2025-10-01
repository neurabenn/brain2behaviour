#!/usr/bin/env python3
import argparse, os, re, subprocess, copy
from pathlib import Path
import pandas as pd
from brain2behaviour.dataset import BrainBehaviorDataset

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")

def get_slurm_script(perm_id: int, worklist_file: Path, opath: Path, array_size: int):
    log_dir = opath.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    template = f"""#!/bin/bash
#SBATCH --job-name=perm{perm_id:04d}Feat
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array=0-{array_size-1}:1
#SBATCH --requeue
#SBATCH -o {log_dir}/perm{perm_id:04d}-%A_%a.out

ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate

WORKLIST="{worklist_file}"

echo "The job id is $SLURM_ARRAY_JOB_ID"
ITERATION=$SLURM_ARRAY_TASK_ID
echo "Processing task $ITERATION"
CMD="$(sed -n "$((ITERATION+1))p" "$WORKLIST")"
echo "Running: $CMD"
eval "$CMD"
"""
    opath.write_text(template)

def submit_array(sbatch_path: Path) -> str:
    res = subprocess.run(["sbatch", "--export=ALL", str(sbatch_path.resolve())],
                         text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"sbatch failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
    m = _JOBID_RE.search(res.stdout)
    if not m:
        raise RuntimeError(f"Could not parse job id from: {res.stdout!r}")
    return m.group(1)

def submit_collect_after(jobid: str, *, dataset_path: str, feature_dir: str, task: str,
                         logs_dir: Path, env_activate: str,
                         python_cmd: str = "python3",
                         script_path: str = "/well/margulies/users/mnk884/DistanceValidationOx2025/brain2behaviour/src/brain2behaviour/cluster_scripts/CollectFeaturesandPredict-SES.py") -> str:
    logs_dir.mkdir(parents=True, exist_ok=True)
    wrap = f'bash -lc "{env_activate}; {python_cmd} {script_path} {dataset_path} {feature_dir} {task}"'
    cmd = ["sbatch", "-J", f"collect_{task}", "-o", str(logs_dir / f"collect-{task}-%j.out"),
           "--dependency", f"afterok:{jobid}", "--wrap", wrap]
    res = subprocess.run(cmd, text=True, capture_output=True)
    print("sbatch STDOUT:", res.stdout.strip())
    print("sbatch STDERR:", res.stderr.strip())
    if res.returncode != 0:
        raise RuntimeError(f"sbatch collect failed with code {res.returncode}")
    m = _JOBID_RE.search(res.stdout)
    if not m:
        raise RuntimeError(f"Could not parse job id from: {res.stdout!r}")
    return m.group(1)

def main():
    ap = argparse.ArgumentParser(description="Launch CPM feature extraction + collect/predict for all permutations.")
    ap.add_argument("--dataset", required=True, help="Path to pickled BrainBehaviorDataset (.pkl)")
    ap.add_argument("--perms",   required=True, help="CSV of permutation indices (subjects x perms, 0-indexed; perms in columns)")
    ap.add_argument("--task",    default="ReadEng_Unadj", help="Behavior column to use for CPM/collect")
    ap.add_argument("--env-activate", default="source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate")
    ap.add_argument("--select-script", default="/well/margulies/users/mnk884/DistanceValidationOx2025/brain2behaviour/src/brain2behaviour/cluster_scripts/select_features_batch.py")
    args = ap.parse_args()

    ds = BrainBehaviorDataset.load(args.dataset)
    out_stem = ds.filepath.split('.')[0]
    out_dir  = Path(f"{out_stem}_cpm_analysis")
    out_dir.mkdir(exist_ok=True)

    perm_df  = pd.read_csv(args.perms, header=None)   # perms in columns, 0-indexed
    perm_mat = perm_df.values                         # shape: (n_subj, n_perms)
    n_perms  = perm_mat.shape[1]

    beh_data = ds.behaviorData.copy()

    # iterate permutations by COLUMN
    for j in range(n_perms):
        out_perm_path = str(out_dir / f"ds_permed{j:04d}.pkl")
        feature_dir   = str(out_dir / f"ds_permed{j:04d}_features_tmp")
        os.makedirs(feature_dir, exist_ok=True)

        permed_ds = copy.deepcopy(ds)
        permed_ds.filepath = out_perm_path
        permed_ds.features = []

        # shuffle values, keep indices (so .loc later won’t undo it)
        idx = perm_mat[:, j].astype(int)
        permed_ds.behaviorData = pd.DataFrame(
            beh_data.iloc[idx].to_numpy(),
            index=beh_data.index, columns=beh_data.columns
        )
        permed_ds.save(overwrite=True)

        # write worklist: one line per fold
        folds = list(permed_ds.cv_folds.keys())
        worklist = Path(feature_dir) / "feature__extraction_worklist.txt"
        with worklist.open("w") as fh:
            for fold in folds:
                fh.write(f"python {args.select_script} --dataset {out_perm_path} --fold {fold} --passthrough_cols SSAGA_Income SSAGA_Educ\n")

        # write sbatch for this permutation (array spans #folds lines)
        sbatch_path = Path(str(worklist).replace("txt", "sh"))
        get_slurm_script(j, worklist, sbatch_path, array_size=len(folds))

        # submit array + dependent collect/predict
        jobid = submit_array(sbatch_path)
        print(f"Submitted array for perm {j:04d}: job {jobid}")
        collect_jobid = submit_collect_after(
            jobid,
            dataset_path=out_perm_path,
            feature_dir=feature_dir,
            task=args.task,
            logs_dir=sbatch_path.parent / "logs",
            env_activate=args.env_activate,
        )
        print(f"Submitted array {jobid} and collect {collect_jobid} for perm {j:04d}")

if __name__ == "__main__":
    main()
