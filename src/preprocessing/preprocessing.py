import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from utils import palm_inormal,normal_eqn_python,zscore,cube_root


#### function to residualize distances by subject confounds.
def preprocessDists(data,confounds):    
    NET=data.copy()
    dims=NET.shape
    ##### check for vertices with no variance i.e guaranteed masks 
    steady_masks=np.where(np.sum(NET)==0)[0]
    valididx=np.where(np.sum(NET)!=0)[0]
    
    if len(steady_masks)!=0:
        NET=NET.iloc[:,valididx]
        
#     amNET = np.abs(np.nanmean(NET, axis=0))
    NET1 = NET#/amNET
    NET1=NET1-np.mean(NET1,axis=0)
    NET1=NET1/np.nanstd(NET1.values.flatten())
    NET1=normal_eqn_python(confounds,NET1)
    NET1=pd.DataFrame(NET1,columns=NET.columns,index=data.index)
    
    if len(steady_masks)!=0:
        out=np.zeros(dims)
        out[:,valididx]=NET1.values
        NET1=pd.DataFrame(out,index=NET.index)
    
    return NET1
    

### confound prep 
class PrepConfounds(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        subject_list=None,
        drop_cols=None,
        encode_cols=None,
        bin_encode=None,
        volume_cols=None,
        area_cols=None,
        gaussianize=True,
        add_squares=True,
        zscore_cols=True,
    ):
        """
        Parameters
        ----------
        subject_list : list-like, optional
            Indices of rows to keep.
        drop_cols : list of str, optional
            Columns to drop before any processing.
        encode_cols : list of str, optional
            Columns to label-encode.
        bin_encode : dict {col: threshold}, optional
            For each col, apply the 2-step binarization.
        volume_cols : list of str, optional
            Cube-root transform these columns.
        area_cols : list of str, optional
            Sqrt transform these columns.
        gaussianize : bool, default=True
            Apply palm_inormal to all features.
        add_squares : bool, default=True
            Append squared features of all but first column.
        zscore_cols : bool, default=True
            Z-score the final matrix.
        """
        self.subject_list = subject_list
        self.drop_cols    = drop_cols or []
        self.encode_cols  = encode_cols or []
        self.bin_encode   = bin_encode or {}
        self.volume_cols  = volume_cols or []
        self.area_cols    = area_cols or []
        self.gaussianize  = gaussianize
        self.add_squares  = add_squares
        self.zscore_cols  = zscore_cols

    def fit(self, X, y=None):
        """
        Learn any data-driven parameters (here: LabelEncoders).
        """
        X = self._validate_input(X)
        self.encoders_ = {
            col: LabelEncoder().fit(X[col])
            for col in self.encode_cols
        }
        return self

    def transform(self, X, y=None):
        """
        Apply the full preprocessing pipeline.
        """
        check_is_fitted(self, "encoders_")
        X_enc = self._validate_input(X).copy()

        # 1) subset rows
        if self.subject_list is not None:
            X_enc = X_enc.loc[self.subject_list]

        # 2) drop columns
        if self.drop_cols:
            X_enc = X_enc.drop(self.drop_cols, axis=1)

        # 3) label-encode
        for col, le in self.encoders_.items():
            X_enc[col] = le.transform(X_enc[col])

        # 4) two-step binary encoding
        for col, thresh in self.bin_encode.items():
            arr = X_enc[col].values.copy()
            arr[arr < thresh] = 0
            arr[arr > 0]      = 1
            X_enc[col]        = arr

        # 5) area/volume transforms
        for col in self.area_cols:
            X_enc[col] = np.sqrt(X_enc[col])
        for col in self.volume_cols:
            X_enc[col] = cube_root(X_enc[col])

        # 6) Gaussianize
        if self.gaussianize:
            cols = X_enc.columns
            arr  = palm_inormal(X_enc)
            X_enc = pd.DataFrame(arr, index=X_enc.index, columns=cols)

        # 7) Add squared features
        if self.add_squares:
            sq = X_enc.iloc[:, 1:] ** 2
            sq.columns = [f"{c}_sq" for c in sq.columns]
            X_enc = pd.concat([X_enc, sq], axis=1)

        # 8) Z-score
        if self.zscore_cols:
            arr   = zscore(X_enc.values, ax=0)
            X_enc = pd.DataFrame(arr, index=X_enc.index, columns=X_enc.columns)

        return X_enc

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
        return X

######## distance prep 
class DistancePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, confounds=None):
        """
        X          : distance DataFrame for the training fold
        confounds  : array‐like or DataFrame of cleaned confounds (training fold)
        """
        if confounds is None:
            raise ValueError("DistancePrep.fit requires confounds")
        # store confounds for use in transform
        self.confounds_ = (
            confounds.values
            if isinstance(confounds, pd.DataFrame)
            else confounds
        )
        return self

    def transform(self, X):
        """
        X : distance DataFrame (train or test subset)
        """
        # use the stored confounds_ for residualization
        return preprocessDists(X, self.confounds_)

#### behavior prep 


class BehaviorPrep(BaseEstimator, TransformerMixin):
    def __init__(self, gaussianize=True):
        self.gaussianize = gaussianize

    def fit(self, X, confounds):
        """
        X          : DataFrame of behavior data (train fold)
        confounds  : array‐like or DataFrame of cleaned confounds (train fold)
        """
        # Store confounds for transform
        if isinstance(confounds, pd.DataFrame):
            self.confounds_ = confounds.values
        else:
            self.confounds_ = confounds
        return self

    def transform(self, X):
        """
        X : DataFrame of behavior data (can be train or test)
        Uses self.confounds_ to residualize.
        """
        # Gaussianize if requested
        beh_arr = palm_inormal(X.values) if self.gaussianize else X.values
        # Residualize
        resid   = normal_eqn_python(self.confounds_, beh_arr)
        # Z-score
        z_arr   = zscore(resid, ax=0)
        return pd.DataFrame(z_arr, index=X.index, columns=X.columns)

