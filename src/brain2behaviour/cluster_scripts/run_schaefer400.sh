#!/bin/bash
#SBATCH -J End2EndCPM
#SBATCH -p long
#SBATCH --cpus-per-task=1
#SBATCH -o logs/End2End-%j.out

ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate



python End2EndCPM_wPerms.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP//Distance/Schaefer400/Reading_DistanceCentroid.pkl --perms=/well/margulies/projects/DistanceValidation/data/HCP/permutations/enhanced_dataset_reading/pset_single_1000perms.csv --task ReadEng_Unadj

python End2EndCPM_wPerms.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP//FC/Schaefer400/Reading_FC.pkl --perms=/well/margulies/projects/DistanceValidation/data/HCP/permutations/enhanced_dataset_reading/pset_single_1000perms.csv --task ReadEng_Unadj

python End2EndCPM_wPerms.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP//SurfaceArea/Schaefer400/Reading_SurfaceArea.pkl --perms=/well/margulies/projects/DistanceValidation/data/HCP/permutations/enhanced_dataset_reading/pset_single_1000perms.csv --task ReadEng_Unadj 
