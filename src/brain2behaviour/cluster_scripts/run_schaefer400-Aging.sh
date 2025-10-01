#!/bin/bash
#SBATCH -J End2EndCPM
#SBATCH -p long
#SBATCH --cpus-per-task=1
#SBATCH -o logs/End2End-%j.out

ml use -a /apps/eb/2020b/skylake/modules/all
module load Python/3.8.2-GCCcore-9.3.0
source /well/margulies/users/mnk884/DistanceValidationOx2025/Environments/MeasureDistancesandFCNew/bin/activate


# ### aging dataset
# ### pearson correlation
# ### centroids used
# python End2EndCPM_wPerms-Aging.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP-A/Distance/Schaefer400/Reading_Distance.pkl --perms='/well/margulies/projects/data/HCP-Aging/PermutationSets/perms.csv' --task ReadEng_Unadj

python End2EndCPM_wPerms-Aging.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP-A/Distance/Schaefer400/Reading_Distance_0.05.pkl --perms='/well/margulies/projects/data/HCP-Aging/PermutationSets/perms.csv' --task ReadEng_Unadj



# ### aging under 65
# python End2EndCPM_wPerms-Aging.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP-A/Distance/Schaefer400/Reading_Distance_U65.pkl --perms='/well/margulies/projects/data/HCP-Aging/PermutationSets/permsU65.csv' --task ReadEng_Unadj

# ### surface area
# python End2EndCPM_wPerms-Aging.py --dataset=/well/margulies/projects/DistanceValidation/data/HCP-A/SurfaceArea/Schaefer400/Reading_SA.pkl --perms='/well/margulies/projects/data/HCP-Aging/PermutationSets/perms.csv' --task ReadEng_Unadj
