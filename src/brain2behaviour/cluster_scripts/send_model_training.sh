#!/bin/bash
#SBATCH --job-name=CPMTrain
#SBATCH -o ./logs/ReadModels-%j-%a.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array=0-15000:1
#SBATCH --requeue
ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate

echo "The job id is $SLURM_ARRAY_JOB_ID"
ITERATION=$SLURM_ARRAY_TASK_ID
echo "Processing fold $ITERATION"

# ----- pick the N-th line (1-based for sed) -----
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" training_worklist.txt)
DATASET=$(awk '{print $1}'    <<< "$line")
FOLD=$(awk    '{print $2}'    <<< "$line")
SIGN=$(awk    '{print $3}'    <<< "$line")
PERM=$(awk    '{print $4}'    <<< "$line")
OUTPATH=$(awk '{print $5}'    <<< "$line")

echo python3 -u run_reading_models.py --dataset $DATASET --fold $FOLD --sign $SIGN --perms $PERM --outpath $OUTPATH
python3 -u run_reading_models.py --dataset $DATASET --fold $FOLD --sign $SIGN --perms $PERM --outpath $OUTPATH



