#!/usr/bin/env python3
import sys,argparse, os, multiprocessing as mp
import typing
import nibabel as nib
import hcp_utils as hcp
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import signal
from typing import List
from surfdist.load import load_cifti_labels,load_gifti_labels

#### utilty functions 

def ZscoreFiltTs(func,v_cens=10,detrend=True,standardize='zscore',filt='butterworth',low=0.08,high=0.008):
	"""Zscore normalize, bandpass filter, and remove first 10 volumes"""
	cifti=nib.load(func)
	#### clean the time series 
	cln=signal.clean(cifti.get_fdata(),detrend=detrend,standardize='zscore',filter=filt,low_pass=low,high_pass=high)
	return cln[v_cens:]

def get_timeseries(data:list):
    timeSeries=np.vstack([ZscoreFiltTs(i) for i in data])
    return timeSeries

def validate_file_list(file_list: List[str]) -> List[str]:
    for f in file_list:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"File not found: {f}")
    return file_list

#### get cifti index function from dtseries files
def get_cifti_ts_info(cifti_file):
    cii=nib.load(cifti_file)
    axes = [cii.header.get_axis(i) for i in range(cii.ndim)]
    bm_axis = axes[1]
    # For left cortex
    lh_n_vertices = bm_axis.nvertices['CIFTI_STRUCTURE_CORTEX_LEFT']
    # For right cortex
    rh_n_vertices = bm_axis.nvertices['CIFTI_STRUCTURE_CORTEX_RIGHT']
    print("Left hemisphere total vertices:", lh_n_vertices)
    print("Right hemisphere total vertices:", rh_n_vertices)
    
    
    # Assume cii is already loaded and axes[1] is BrainModelAxis
    
    # Initialize containers
    lh_vertex_indices = []
    rh_vertex_indices = []
    
    # # Loop through brain models
    for bm in bm_axis:
        if bm[2] == "CIFTI_STRUCTURE_CORTEX_LEFT":
            lh_vertex_indices.append(bm[1])
        elif bm[2] == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            rh_vertex_indices.append(bm[1])
    
    # # # Convert to numpy arrays
    lh_vertex_indices = np.array(lh_vertex_indices)
    
    rh_vertex_indices = np.array(rh_vertex_indices)
    print("Left hemisphere cortical vertices:", len(lh_vertex_indices))
    print("Right hemisphere cortical vertices:", len(rh_vertex_indices))
    
    return {'l_nverts':lh_n_vertices,'r_nverts':rh_n_vertices,'lh_cort_verts':lh_vertex_indices,'rh_cort_verts':rh_vertex_indices}



##### FC matrix calculation using CIFTI labels

def corrmatCiftiLabel(time_series,cifti_info,ciftiLabel):
    #### takes the hcp class as input already. let's us only generate the time series once 
    print('doing left hemisphere') 
    ### set up empty matrix to fill 
    print(time_series.shape[0])
    fullctL=np.zeros([time_series.shape[0],cifti_info['l_nverts']])
    print(fullctL.shape)
    
    fullctL[:,cifti_info['lh_cort_verts']]=time_series[:,0:cifti_info['lh_cort_verts'].shape[0]]
    
    labelsL=load_cifti_labels(ciftiLabel,'L')
    del labelsL['???']
    funcL={}
    for key in labelsL:
        funcL[key]=np.mean(fullctL[:,labelsL[key]],axis=1)
    
    funcL_df = pd.DataFrame.from_dict(funcL).T
    print(funcL_df.shape)
    corrmat_L=np.corrcoef(funcL_df)
    upperL=np.triu_indices(corrmat_L.shape[0],k=1)
    upperL=corrmat_L[upperL]
    
    print('doing the right hemisphere')
    fullctR=np.zeros([time_series.shape[0],cifti_info['r_nverts']])
    print(fullctR.shape)
    fullctR[:,cifti_info['rh_cort_verts']]=time_series[:,0:cifti_info['rh_cort_verts'].shape[0]]
    
    labelsR=load_cifti_labels(ciftiLabel,'R')
    del labelsR['???']
    funcR={}
    for key in labelsR:
        funcR[key]=np.mean(fullctR[:,labelsR[key]],axis=1)
        
    funcR_df = pd.DataFrame.from_dict(funcR).T
    corrmat_R=np.corrcoef(funcR_df)
    upperR=np.triu_indices(corrmat_R.shape[0],k=1)
    upperR=corrmat_R[upperR]
    
    vectorized_single_hemi=np.concatenate([upperL,upperR])
    print(f'shape of vectorized hemi is {vectorized_single_hemi.shape}')

    #### run the cross hemisphere conenctivity too. 
    combined_ts = pd.concat([funcL_df, funcR_df], axis=0)
    print(combined_ts.shape)
    corr=np.corrcoef(combined_ts)
    upper=np.triu_indices(corr.shape[0],k=1)
    vectorized_cross_hemi=corr[upper]
    

    return vectorized_single_hemi,vectorized_cross_hemi

#### FC matrix calculation using gifti labels

def corrmatGiftiLabels(time_series,cifti_info,lh_gifti,rh_gifti):
    #### takes the hcp class as input already. let's us only generate the time series once 
    print('doing left hemisphere') 
    ### set up empty matrix to fill 
    print(time_series.shape[0])
    fullctL=np.zeros([time_series.shape[0],cifti_info['l_nverts']])
    print(fullctL.shape)
    
    fullctL[:,cifti_info['lh_cort_verts']]=time_series[:,0:cifti_info['lh_cort_verts'].shape[0]]
    
    labelsL=load_gifti_labels(lh_gifti)
    del labelsL['???']
    funcL={}
    for key in labelsL:
        funcL[key]=np.mean(fullctL[:,labelsL[key]],axis=1)
    
    funcL_df = pd.DataFrame.from_dict(funcL).T
    print(funcL_df.shape)
    corrmat_L=np.corrcoef(funcL_df)
    upperL=np.triu_indices(corrmat_L.shape[0],k=1)
    upperL=corrmat_L[upperL]
    
    print('doing the right hemisphere')
    fullctR=np.zeros([time_series.shape[0],cifti_info['r_nverts']])
    print(fullctR.shape)
    fullctR[:,cifti_info['rh_cort_verts']]=time_series[:,0:cifti_info['rh_cort_verts'].shape[0]]
    
    labelsR=load_gifti_labels(rh_gifti)
    del labelsR['???']
    funcR={}
    for key in labelsR:
        funcR[key]=np.mean(fullctR[:,labelsR[key]],axis=1)
        
    funcR_df = pd.DataFrame.from_dict(funcR).T
    corrmat_R=np.corrcoef(funcR_df)
    upperR=np.triu_indices(corrmat_R.shape[0],k=1)
    upperR=corrmat_R[upperR]
    
    vectorized_single_hemi=np.concatenate([upperL,upperR])
    print(f'shape of vectorized hemi is {vectorized_single_hemi.shape}')

    #### run the cross hemisphere conenctivity too. 
    combined_ts = pd.concat([funcL_df, funcR_df], axis=0)
    print(combined_ts.shape)
    corr=np.corrcoef(combined_ts)
    upper=np.triu_indices(corr.shape[0],k=1)
    vectorized_cross_hemi=corr[upper]
    print(f'shape of vectorized cross hemi is {vectorized_cross_hemi.shape}')
    

    return vectorized_single_hemi,vectorized_cross_hemi





# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure distances with a given parcellation for a single subject"
    )
    parser.add_argument(
        "--time_series", "-ts",
        type=str,
        required=True,
        nargs='+',  # Accepts one or more arguments
        help="Path(s) to one or more cifti time series files"
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
    ts=args.time_series
    print(type(ts))
    cifti_parc=args.cifti_parc
    lh_gifti=args.lh_gifti
    rh_gifti=args.rh_gifti
    outdir=args.outdir
    
    ### create the out directory if it doesn't exist already 
    os.makedirs(outdir, exist_ok=True)
    prefix=args.prefix
    opath=f'{outdir}/{prefix}_FC.npy'
    #### get number of cpus 

    # Example: you can access arguments like so:
    print(f"Timeseries files : {ts}")
    print(f"CIFTI parcellation: {cifti_parc}")
    print(f"Left GIFTI: {lh_gifti}")
    print(f"Right GIFTI: {rh_gifti}")
    print(f"Output path: {opath}")
    
    ### run for cifti
    ts_vertex_info=get_cifti_ts_info(ts[0])
    if cifti_parc:
        print('yay cifti parcellation')
        ts=get_timeseries(ts)
        vectorized_FC_byHemi,vectorizedFC_crossHemi=corrmatCiftiLabel(ts,ts_vertex_info,cifti_parc)
        np.save(opath,vectorized_FC_byHemi)
        np.save(f'{outdir}/{prefix}_FC_crossHemi.npy',vectorizedFC_crossHemi)

    if lh_gifti and rh_gifti:
        print('Running GIFTI mode')
        ts=get_timeseries(ts)
        vectorized_FC_byHemi,vectorizedFC_crossHemi=corrmatGiftiLabels(ts,ts_vertex_info,lh_gifti,rh_gifti)
        np.save(opath,vectorized_FC_byHemi)
        np.save(f'{outdir}/{prefix}_FC_crossHemi.npy',vectorizedFC_crossHemi)


if __name__ == "__main__":
    main()
