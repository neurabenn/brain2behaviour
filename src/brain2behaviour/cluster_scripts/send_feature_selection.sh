#!/bin/bash
#SBATCH --job-name=CPMFeat
#SBATCH -o ./logs/FeatureExtraction-%j-%a.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=3
#SBATCH --array=0-5000
#SBATCH --requeue
ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate


echo "The job id is $SLURM_ARRAY_JOB_ID"
ITERATION=$SLURM_ARRAY_TASK_ID
echo "Processing fold $ITERATION"

# ----- pick the N-th line (1-based for sed) -----
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" worklist.txt)
DATASET=$(awk '{print $1}' <<< "$line")
echo $DATASET
FOLD=$(awk   '{print $2}' <<< "$line")
echo $FOLD

echo python3 -u select_features_batch.py --dataset $DATASET --fold $FOLD
python3 -u select_features_batch.py --dataset $DATASET --fold $FOLD



