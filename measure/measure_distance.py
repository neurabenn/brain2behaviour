#!/usr/bin/env python3
import sys,argparse, os, multiprocessing as mp
import numpy as np
from pathlib import Path
from surfdist.analysis import dist_calc_matrix
from surfdist.load import load_cifti_labels

def measure_cifti_dist(surfaceL,surfaceR,parcellation,centroid,n_cpus=1):
    print(f'calcultating distance matrices with {n_cpus} cores')
    surfaces={'L':surfaceL,'R':surfaceR}  
    dvecs=[]
    for hemi in surfaces:
        print(f'Measuring distances from {hemi} hemisphere')
        dmat,rois=dist_calc_matrix(surfaces[hemi],parcellation,hemi,n_cpus=n_cpus,centroid=centroid)
        upper=np.triu_indices(dmat.shape[0],k=1)
        dmat=dmat[upper]
        print(dmat.shape)
        print(f'{hemi} hemisphere has {len(dmat)} edges')
        dvecs.append(dmat)
    distance_vector=np.concatenate(dvecs)
    return distance_vector
        

def measure_gifti_dist(surfaceL,surfaceR,parcelL,parcelR,centroid,n_cpus=1):
    dmatL,rois=dist_calc_matrix(surfaceL,parcelL,'L',n_cpus=n_cpus,centroid=centroid)
    upperL=np.triu_indices(dmatL.shape[0],k=1)

    dmatR,rois=dist_calc_matrix(surfaceR,parcelR,'R',n_cpus=n_cpus,centroid=centroid)
    upperR=np.triu_indices(dmatR.shape[0],k=1)

    dmatL=dmatL[upperL]
    print(f'Left hemisphere has {dmatL.shape} edges')
    dmatR=dmatR[upperR]
    print(f'Right hemisphere has {dmatR.shape} edges')

    distance_vector=np.concatenate([dmatL,dmatR])
    return distance_vector


    



# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure distances with a given parcellation for a single subject"
    )
    parser.add_argument(
        "--left_hemisphere", "-lh",
        type=str, required=True,
        help="Path to left hemisphere surface file"
    )
    parser.add_argument(
        "--right_hemisphere", "-rh",
        type=str, required=True,
        help="Path to right hemisphere surface file"
    )
    parser.add_argument(
        "--cifti_parc", "-cifti",
        type=str, required=False,
        help="Path to CIFTI parcellation file"
    )
    parser.add_argument(
        "--lh_gifti", "-lg",
        type=str, required=False,
        help="Path to left hemisphere GIFTI parcellation"
    )
    parser.add_argument(
        "--rh_gifti", "-rg",
        type=str, required=False,
        help="Path to right hemisphere GIFTI parcellation"
    )
    parser.add_argument(
        "--outdir", "-od",
        type=str, required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--prefix", "-p",
        type=str, required=True,
        help="Output prefix"
    )

    parser.add_argument(
        "--centroid", "-c",
        action="store_true",
        help="Use centroid mode (default: False). \
        Will measure distances from parcel centroid instead of parcel borders "
    )
    args = parser.parse_args()

    # --- Logic: If --cifti not provided, then both --lh_gifti and --rh_gifti must be
    if args.cifti_parc is None:
        if args.lh_gifti is None or args.rh_gifti is None:
            parser.error(
                "If --cifti_parc is not provided, both --lh_gifti and --rh_gifti are required.")
    return args


def main() -> None:
    args = parse_args()
    ### assign parsed args
    lsrf=args.left_hemisphere
    rsrf=args.right_hemisphere
    cifti=args.cifti_parc
    lh_gifti=args.lh_gifti
    rh_gifti=args.rh_gifti
    outdir=args.outdir
    centroid=args.centroid
    
    ### create the out directory if it doesn't exist already 
    os.makedirs(outdir, exist_ok=True)
    prefix=args.prefix
    opath=f'{outdir}/{prefix}_distance.npy'
    #### get number of cpus 
    sys_cpus=mp.cpu_count()

    # Example: you can access arguments like so:
    print(f"Left hemisphere surface: {lsrf}")
    print(f"Right hemisphere surface: {rsrf}")
    print(f"CIFTI parcellation: {cifti}")
    print(f"Left GIFTI: {lh_gifti}")
    print(f"Right GIFTI: {rh_gifti}")
    print(f"Output path: {opath}")
    
    ##3 run for cifti
    if cifti:
        distance_vec=measure_cifti_dist(lsrf,rsrf,cifti,centroid,n_cpus=sys_cpus)
        np.save(opath,distance_vec)

    if lh_gifti and rh_gifti:
        print('Running GIFTI mode')
        distance_vec=measure_gifti_dist(lsrf,rsrf,lh_gifti,rh_gifti,centroid,n_cpus=sys_cpus)
        np.save(opath,distance_vec)



if __name__ == "__main__":
    main()
