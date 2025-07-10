import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
import multiprocessing
from multiprocessing import Pool, cpu_count

def calc_corr_by_col(brain, beh, method='spearman'):
    r_vals = {}
    p_vals = {}
    for measure in beh:
        r_col = {}
        p_col = {}
        for edge in brain:
            if method == 'spearman':
                r, p = spearmanr(brain[edge], beh[measure])
            else:
                r, p = pearsonr(brain[edge], beh[measure])
            r_col[edge] = r
            p_col[edge] = p
        r_vals[measure] = r_col
        p_vals[measure] = p_col

    # DataFrames with edges as index, measures as columns
    df_r = pd.DataFrame(r_vals)
    df_p = pd.DataFrame(p_vals)

    # MultiIndex columns
    arrays = []
    for measure in beh:
        arrays.append((measure, 'r'))
        arrays.append((measure, 'p'))
    tuples = arrays
    multi_cols = pd.MultiIndex.from_tuples(tuples, names=["measure", "stat"])

    # Ensure DataFrames not Series
    concat_cols = []
    for measure in beh:
        concat_cols.append(df_r[[measure]])  # DataFrame
        concat_cols.append(df_p[[measure]])  # DataFrame
    df_out = pd.concat(concat_cols, axis=1)
    df_out.columns = multi_cols

    # # Optionally, reindex to preserve brain.columns order and beh.columns order
    df_out = df_out.reindex(index=brain.columns)

    return df_out.T

def filter_r_by_p(df, p_thresh=0.01):
    """
    For each measure in the index, return r values for columns where p < p_thresh.
    Assumes: df is a DataFrame with MultiIndex index (measure, stat), 
    and columns are edges (or features).
    """
    result = {}
    measures = df.index.get_level_values(0).unique()
    for measure in measures:
        # Get DataFrame for this measure (rows: ['r','p'], columns: edges)
        subdf = df.loc[measure]
        # subdf has index ['r','p'], columns are the edges/features
        pvals = subdf.loc['p']    # Series: columns/edges as index, p values as values
        mask = pvals < p_thresh
        # Keep only columns with p < threshold, then select 'r' row
        rvals = subdf.loc['r', mask]
        result[measure] = rvals
    return result

# Helper to make each batch a tuple for pool.starmap
def batch_spearman(args):
    batch_df, beh = args
    return calc_corr_by_col(batch_df, beh,)


# Helper to make each batch a tuple for pool.starmap
def batch_pearson(args):
    batch_df, beh = args
    return calc_corr_by_col(batch_df, beh,method='pearson')

def get_CPM_features(CleanedData, pthresh=0.01, batch_size=1, ncpus=1,method='spearman'):
    print(f'using {method}')
    brain = CleanedData['BrainTrainClean']
    beh = CleanedData['BehTrainClean']
    
    if batch_size == 1 or ncpus == 1:
        # Serial processing
        feature_p = calc_corr_by_col(brain, beh,method=method)
        return filter_r_by_p(feature_p, pthresh)
    else:
        print('running in parallel')
        n_edges = brain.shape[1]
        n_batches = int(np.ceil(n_edges / batch_size))
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_edges)
            batch_cols = brain.columns[start:end]
            batch_df = brain.loc[:, batch_cols]
            batches.append((batch_df, beh))
        # Use ncpus or maximum available
        n_workers = min(ncpus, cpu_count())
        if method=='spearman':
            with Pool(processes=n_workers) as pool:
                results = pool.map(batch_spearman, batches)
        else:
            with Pool(processes=n_workers) as pool:
                    results = pool.map(batch_pearson, batches)
        # results is a list of DataFrames (batches of p-values)
        feature_p = pd.concat(results, axis=1)
        return filter_r_by_p(feature_p, pthresh)
