import numpy as np
import pandas as pd
import pickle

##### helper function
def create_cv_groups(groups, groupedData):
    """
    Create K-fold splits on families.

    :param groups: List of arrays, each containing family IDs for one fold.
    :param groupedData: pandas GroupBy object on 'Family_ID'.
    :return: dict mapping fold names to {'training': [...], 'testing': [...]}.
    """
    cv_folds = {}
    for i, fam_block in enumerate(groups):
        # Build test indices
        test_idx = pd.concat([groupedData.get_group(fam) for fam in fam_block]).index.tolist()
        # Build training indices (all other families)
        train_fams = [fam for j, grp in enumerate(groups) if j != i for fam in grp]
        train_idx = pd.concat([groupedData.get_group(fam) for fam in train_fams]).index.tolist()
        fold_key = f"fold{i + 1}"
        cv_folds[fold_key] = {'training': train_idx, 'testing': test_idx}
    return cv_folds



def cube_root(x):
    return x**(1/3)

def normal_eqn_python(X,Y):
    X=np.asarray(X,dtype='float32')
    Y=np.asarray(Y,dtype='float32')
    params=np.linalg.pinv(np.dot(X.T,X)).dot(X.T).dot(Y)
    resid=Y-np.dot(params.T,X.T).T
    return resid

def zscore(x,ax=0):
    """z normalize on the first or second axis of the data set"""
    if ax==0:
        # print('column wise normalization')
        z=(x-np.nanmean(x,axis=0))/np.nanstd(x,axis=0)
    elif ax==1:
        # print('row wise normalization')
        z=(x.T-np.nanmean(x,axis=1))/np.nanstd(x,axis=1)
    
    z[~np.isfinite(z)]=0
    return z


import numpy as np
from scipy.special import erfinv

def palm_inormal(X, c=None, method=None, quanti=False):
    """
    This function is a port of inormal in Anderson Winkler's PALM package
    https://github.com/andersonwinkler/PALM
    
    Apply a rank-based inverse normal transformation to the data.

    Parameters
    ----------
    X : array-like, shape (n_samples,) or (n_samples, n_features)
        Input data. Can be a 1D or 2D array.
    c : float, optional
        Constant in the transformation formula. Default is 3/8 (Blom).
    method : {'Blom', 'Tukey', 'Bliss', 'Waerden', 'SOLAR'}, optional
        Predefined method to set c. If provided, overrides c:
            - 'Blom'   -> 3/8
            - 'Tukey'  -> 1/3
            - 'Bliss'  -> 1/2
            - 'Waerden' or 'SOLAR' -> 0
    quanti : bool, default=False
        If True, assumes X has no NaNs and is quantitative. Performs a faster, column-wise transform
        without handling missing values or ties.

    Returns
    -------
    Z : ndarray, same shape as X
        Transformed data, following the inverse normal (probit) scale.
    """
    # Determine the constant c
    if method is not None:
        m = method.lower()
        methods = {'blom': 3/8, 'tukey': 1/3, 'bliss': 1/2, 'waerden': 0, 'solar': 0}
        if m not in methods:
            raise ValueError(f"Method '{method}' unknown. Use 'Blom', 'Tukey', 'Bliss', 'Waerden' or 'SOLAR'.")
        c_val = methods[m]
    else:
        c_val = 3/8 if c is None else c

    X_arr = np.asarray(X, dtype=float)
    # Ensure 2D for unified processing
    orig_shape = X_arr.shape
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n_samples, n_features = X_arr.shape
    Z = np.full_like(X_arr, np.nan, dtype=float)

    if quanti:
        # Fast column-wise transform, no NaN or tie handling
        # Compute rank (1-based) for each column
        ranks = np.argsort(np.argsort(X_arr, axis=0), axis=0) + 1  # shape (n_samples, n_features)
        p = (ranks - c_val) / (n_samples - 2*c_val + 1)
        Z = np.sqrt(2) * erfinv(2*p - 1)
    else:
        # Handle each column separately, accounting for NaNs and ties
        for j in range(n_features):
            col = X_arr[:, j]
            mask = ~np.isnan(col)
            if not np.any(mask):
                # all NaNs remain
                continue
            x = col[mask]
            # Compute rank (1-based)
            ri = np.argsort(np.argsort(x)) + 1
            N = x.size
            p = (ri - c_val) / (N - 2*c_val + 1)
            y = np.sqrt(2) * erfinv(2*p - 1)

            # Average over ties
            unique_vals, inv = np.unique(x, return_inverse=True)
            if unique_vals.size < N:
                # There are ties
                for ui in range(unique_vals.size):
                    tie_idx = (inv == ui)
                    if np.sum(tie_idx) > 1:
                        y[tie_idx] = y[tie_idx].mean()

            Z[mask, j] = y

    # Restore original shape
    if orig_shape and len(orig_shape) == 1:
        return Z.ravel()
    return Z.reshape(orig_shape)


