import copy
import numpy as np
import pandas as pd
import pickle
from collections.abc import Sequence
from typing import Callable,Optional
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.utils import shuffle
from scipy.stats import pearsonr
from brain2behaviour.dataset import BrainBehaviorDataset
from brain2behaviour.preprocessing import clean_fold

# helper to check “all items equal the first -- used to ensure subjects are consistent across datasets”
def all_equal(seq_list):
    return all(seq == seq_list[0] for seq in seq_list)
#### add in an ablation like framework here too later on 
def sum_features4CPM(dataset,sign2keep=None):
    for task in dataset.features:
        sum_feats={}
        for sign in dataset.features[task]:
            sum_feats[sign]=np.sum(dataset.brainData[list(dataset.features[task][sign])],axis=1)
        if sign2keep=='negative':
            del sum_feats['positive']
        elif sign2keep=='positive':
            del sum_feats['negative']                             
        return pd.DataFrame(sum_feats)
        
def prep_ds_CPM_classic(
    datasets,
    fold,
    sign2keep,
    encode_cols,
    bin_encode,
    area_cols,
    volume_cols,
    gaussianize=True,
    add_squares=True,
    zscore_cols=True
):
    """
    Prepare one or more BrainBehaviorDataset instances at a specific fold for
    classic CPM feature extraction and downstream modeling.

    Parameters
    ----------
    datasets : BrainBehaviorDataset or sequence of BrainBehaviorDataset
        A dataset instance or a list/tuple of them. Must each have a
        `.cv_folds` dict containing the given `fold`.
    fold : str
        Fold identifier (e.g. "fold001"); must exist as a key in each
        dataset.cv_folds and map to {'training': [...], 'testing': [...]}.
    sign2keep : {'positive', 'negative'}
        Which sign of CPM features to include in the summary step.
    encode_cols : sequence of str
        Column names to one-hot or ordinal‐encode during preprocessing.
    bin_encode : dict[str, int]
        Mapping of column names → number of bins for discretization.
    area_cols : sequence of str
        Names of columns representing surface areas to include.
    volume_cols : sequence of str
        Names of columns representing volumes to include.
    gaussianize : bool, default=True
        Whether to apply a Gaussianization step to continuous features.
    add_squares : bool, default=True
        Whether to augment with squared terms of continuous features.
    zscore_cols : bool, default=True
        Whether to z-score the specified continuous columns.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dict with the following keys, each a DataFrame aligned on subjects:
        - 'BrainTrainClean':  concatenated brain-feature matrix for training
        - 'BrainTestClean':   concatenated brain-feature matrix for testing
        - 'BehTrainClean':    behavior vector/matrix for training
        - 'BehTestClean':     behavior vector/matrix for testing

    Raises
    ------
    ValueError
        If the training or testing indices differ across datasets for the
        specified fold.
    """
    # Normalize single → list
    if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
        datasets = list(datasets)
    else:
        datasets = [datasets]

    # Check that each dataset has the same train/test for this fold
    fold_dicts = [ds.cv_folds[fold] for ds in datasets]
    train_seqs = [tuple(fd["training"]) for fd in fold_dicts]
    test_seqs  = [tuple(fd["testing"])  for fd in fold_dicts]
    if not all_equal(train_seqs):
        raise ValueError(f"Training indices differ across datasets for fold {fold}")
    if not all_equal(test_seqs):
        raise ValueError(f"Testing indices differ across datasets for fold {fold}")

    # Deep‐copy so originals are untouched
    copies = [copy.deepcopy(ds) for ds in datasets]

    # Summarize CPM features (pre-clean)
    for ds in copies:
        ds.brainData = sum_features4CPM(ds, sign2keep=sign2keep)

    # Clean each copy for the fold
    cpm_clean = {}
    for idx, ds in enumerate(copies, start=1):
        cleaned = clean_fold(
            ds, fold,
            encode_cols=encode_cols,
            bin_encode=bin_encode,
            area_cols=area_cols,
            volume_cols=volume_cols,
            gaussianize=gaussianize,
            add_squares=add_squares,
            zscore_cols=zscore_cols,
        )
        cpm_clean[idx] = cleaned

    # Concatenate brain features (behavior is the same across datasets)
    if len(copies) > 1:
        brain_train = pd.concat(
            [cpm_clean[i]['BrainTrainClean'] for i in cpm_clean],
            axis=1
        )
        brain_test  = pd.concat(
            [cpm_clean[i]['BrainTestClean'] for i in cpm_clean],
            axis=1
        )
    else:
        print(cpm_clean[1]['BrainTrainClean'].shape)
        brain_train = cpm_clean[1]['BrainTrainClean']
        brain_test  = cpm_clean[1]['BrainTestClean']

    # Behavior (take from the first copy as its constant)
    beh_train = cpm_clean[1]['BehTrainClean']
    beh_test  = cpm_clean[1]['BehTestClean']


    
    return {
        'BrainTrainClean':  brain_train,
        'BrainTestClean':   brain_test,
        'BehTrainClean':    beh_train,
        'BehTestClean':     beh_test,
    }

def train_predict_test(
    cleaned_cpm_data,
    model: Optional[RegressorMixin] = None,
    scorer: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,save_model=False):
    """
    Train a regression model on CPM-cleaned training data and evaluate on test data.

    Parameters
    ----------
    cleaned_cpm_data : dict[str, pd.DataFrame]
        Dictionary containing the following keys:
        - 'BrainTrainClean':  DataFrame or array-like, shape (n_train, n_features)
        - 'BehTrainClean':   DataFrame or array-like, shape (n_train,) or (n_train, 1)
        - 'BrainTestClean':   DataFrame or array-like, shape (n_test, n_features)
        - 'BehTestClean':    DataFrame or array-like, shape (n_test,) or (n_test, 1)
    model : RegressorMixin, default=None
        A scikit-learn–style regressor implementing `.fit(X, y)` and `.predict(X)`.
        If None, uses `sklearn.linear_model.LinearRegression()`.
    scorer : callable, default=None
        A function `scorer(y_true, y_pred) -> float` that returns a scalar score.
        If None, defaults to Pearson correlation coefficient (r) via `scipy.stats.pearsonr`.

    Returns
    -------
    score : float
        The score returned by `scorer(y_true, y_pred)`. For the default scorer,
        this is the Pearson r between predictions and true values.

    Examples
    --------
    >>> data = {
    ...     'BrainTrainClean': X_train,
    ...     'BehTrainClean':  y_train,
    ...     'BrainTestClean': X_test,
    ...     'BehTestClean':   y_test,
    ... }
    >>> # default usage: Pearson r with LinearRegression
    >>> r = train_predict_test(data)
    >>> # use a different model and scorer
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.metrics import mean_squared_error
    >>> rmse_scorer = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    >>> score = train_predict_test(data, model=RandomForestRegressor(), scorer=rmse_scorer)
    """
    # 1) Set up model
    if model is None:
        model = LinearRegression()

    # 2) Set up default scorer (Pearson r)
    if scorer is None:
        def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            r, _ = pearsonr(y_true.ravel(), y_pred.ravel())
            return r

    # 3) Extract data
    X_train = cleaned_cpm_data['BrainTrainClean']
    y_train = cleaned_cpm_data['BehTrainClean']
    X_test  = cleaned_cpm_data['BrainTestClean']
    y_test  = cleaned_cpm_data['BehTestClean']

    # 4) Fit & predict
    # Ensure y is a 1D array for sklearn
    y_train_arr = np.asarray(y_train).ravel()
    y_test_arr  = np.asarray(y_test).ravel()

    
    
    model.fit(X_train, y_train_arr)
    y_pred = model.predict(X_test)

    # 5) Score
    if save_model:
        return model,scorer(y_test_arr, y_pred)
    else:
        return scorer(y_test_arr, y_pred)

def evaluate_fold_cpm(clean_data_dict,outpath,fold,perm_set):
    model,r = train_predict_test(clean_data_dict,save_model=True)
    print(f'saving actual model to {outpath}.pkl')
    with open(f'{outpath}/{fold}_model.pkl','wb') as f:
        pickle.dump(model,f)
    if type(perm_set)==int:
        rvals=[]
        r = train_predict_test(clean_data_dict,save_model=False)
        rvals.append(r) ### ensures first entry is actual r value 
        for i in range(perm_set):
            permed_dict=clean_data_dict.copy()
            permed_dict['BehTrainClean']=shuffle(permed_dict['BehTrainClean'])
            r = train_predict_test(permed_dict,save_model=False)
            r_perm.append(r)
    elif type(perm_set)==str:
        perm_set=pd.read_csv(perm_set,header=None)
        rvals=[]
        for i in perm_set:
            ### if using csv then first set should be original order
            permed_dict=clean_data_dict.copy()
            permed_dict['BehTrainClean']=permed_dict['BehTrainClean'].iloc[perm_set[i]]
            r = train_predict_test(permed_dict,save_model=False)
            rvals.append(r)
    out_array=np.asarray(rvals)
    np.save(f'{outpath}/{fold}permuted_r.npy',out_array)
