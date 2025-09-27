#!/bin/bash
#SBATCH --job-name=DistMeasure
#SBATCH -o ./logs/Dists-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=2
#SBATCH --array=1-1018:1
#SBATCH --requeue
#SBATCH -o ./logs/Dists-%A.out
#SBATCH --open-mode=append

worklist=workList_distance_glasser.txt
ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate


echo the job id is $SLURM_ARRAY_JOB_ID
task=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $worklist)
eval "$task"
