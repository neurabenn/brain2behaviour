#!/bin/bash
#SBATCH --job-name=AgeDist
#SBATCH -o ./logs/AgeDists-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array=0-7250:1
#SBATCH --requeue

worklist=workList_HCP_aging.txt
ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate


echo the job id is $SLURM_ARRAY_JOB_ID
task=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $worklist)
eval "$task"
