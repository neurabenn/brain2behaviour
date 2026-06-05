import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from numpy.polynomial.legendre import legval
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from .utils import *
from .transformers import * 

### add option to select hyper parameter fold or main cv folds
class BehPipeline:
    """Simple orchestrator: (y, conf) -> y_clean with steps: gauss -> resid -> z."""
    def __init__(self, gaussianize=True):
        self.gauss = GaussianizeY(enabled=gaussianize)
        self.resid = ResidualizeY()
        self.std   = StandardizeY()
        self.columns_ = None

    def fit(self, y_train, conf_train):
        if isinstance(y_train, pd.Series): y_train = y_train.to_frame()
        self.columns_ = list(y_train.columns)
        Yg   = self.gauss.fit(y_train).transform(y_train)
        Yres = self.resid.fit(conf_train, Yg).transform(conf_train, Yg)
        self.std.fit(Yres)
        return self

    def transform(self, y, conf):
        if isinstance(y, pd.Series): y = y.to_frame()
        Yg   = self.gauss.transform(y)
        Yres = self.resid.transform(conf, Yg)
        Yz   = self.std.transform(Yres)
        return Yz[self.columns_]  # preserve order

    def fit_transform(self, y_train, conf_train):
        self.fit(y_train, conf_train)
        return self.transform(y_train, conf_train)

class BrainPipeline:
    """Order: standardize (fit on train) -> residualize (fit on train)."""
    def __init__(self):
        self.std = StandardizeY()
        self.res = ResidualizeY()
        self.columns_ = None

    def fit(self, brain_train, conf_train):
        if isinstance(brain_train, pd.Series): brain_train = brain_train.to_frame()
        self.columns_ = list(brain_train.columns)
        Bz_train = self.std.fit(brain_train).transform(brain_train)
        self.res.fit(conf_train, Bz_train)
        return self

    def transform(self, brain, conf):
        if isinstance(brain, pd.Series): brain = brain.to_frame()
        Bz = self.std.transform(brain)
        Bres = self.res.transform(conf, Bz)
        return Bres[self.columns_]

    def fit_transform(self, brain_train, conf_train):
        self.fit(brain_train, conf_train)
        return self.transform(brain_train, conf_train)
### go back and make into proper transformer at somepoint
### legendre polynomials for age confounds
### enables us to scale age in a manner which maintains its properties but allows us to compare  to older datasets
### generous range of 18-100 covers everything
def add_age_legendre(df, low=18, high=100):
    t = 2*(df["Age_in_Yrs"] - low)/(high - low) - 1
    L1 = legval(t, [0, 1])      # P1(t) = t
    L2 = legval(t, [0, 0, 1])   # P2(t) = (3t^2 - 1)/2
    return df.assign(Age_L1=L1, Age_L2=L2)

def clean_fold(dataset,fold,encode_cols,area_cols,volume_cols,bin_encode=False,passthrough_cols=None,
               gaussianize = True,add_squares = True,zscore_cols=True,drop_cols=None,pca_head_size=True,hyperparameter=False):
    """Clean a single fold of data. Uses SKlearn pipelines -- see transformers.py for more details"""
    print(f'Cleaning {fold}')
    if hyperparameter==False:
        trainSubjs=dataset.cv_folds[fold]['training']
        testSubjs=dataset.cv_folds[fold]['testing']
    else:
        trainSubjs=list(dataset.hyper_parameterSplits[fold]['training'])
        testSubjs=list(dataset.hyper_parameterSplits[fold]['testing'])
    if passthrough_cols==None:
        passthrough=[]
    else:
        print(f'{passthrough_cols} are being passed through as is')
        passthrough=passthrough_cols
    
    num_cols = [c for c in dataset.confounds.columns if c not in encode_cols]

    exclusion_cols=['Age_in_Yrs']+list(encode_cols)+list(passthrough)
    size_cols = ["FS_IntraCranial_Vol","FS_BrainSeg_Vol","Larea","Rarea"]

    no_gauss=exclusion_cols+size_cols
    print(no_gauss)
    #### confound prep logic
    steps = [
        ("label_enc", MultiColumnLabelEncoder(columns=encode_cols)),
        ("cbrt", CubeRootTransformer(columns=volume_cols)),
        ("sqrt", SquareRootTransformer(columns=area_cols)),]
    if bin_encode:
        for key in bin_encode.keys():
            steps.append((f"binarize_{key}", BinarizeColumnTransformer(column=[key], threshold=bin_encode[key])))
        encode_cols=list(set(list(bin_encode.keys())+list(encode_cols))) ### ensures that binary encoded variables are not further transformed
    if gaussianize:
        steps.append(("inormal", PalmInormalTransformer(
        exclude_cols=no_gauss,
        method="Blom",overwrite=True)))
    if zscore_cols:
        steps.append(("zscore", StandardScalerDF(exclude_cols=encode_cols+passthrough+['Age_in_Yrs'], overwrite=True)))
    if pca_head_size:
        steps.append(("size_pca", PCASizeDF(cols=size_cols, n_components=2,
                                        prefix="SizePC", drop_original=True)))
    if add_squares:
        steps.append(("age_legendre", FunctionTransformer(add_age_legendre, validate=False)))
        steps.append(("drop_age_raw", FunctionTransformer(lambda df: df.drop(columns=["Age_in_Yrs"]), validate=False)))
        # steps.append(("age_decades", FunctionTransformer(lambda df: df.assign(Age_in_Yrs=(df["Age_in_Yrs"] - 50)/10), validate=False)))
        # steps.append(("square_age", SquareTransformer(columns=["Age_in_Yrs"], overwrite=False)))
        # steps.append(("zscore_age_sq", StandardScalerDF(columns=["Age_in_Yrs_sq"], overwrite=True)))
    
    ConfoundPipeline = Pipeline(steps)

    ConfoundPipeline.fit(dataset.confounds.loc[trainSubjs])

    #### the confound pipeline is set so now let's clean the fold
    ### we already have our subjects set up from cv vs hyperparameter selection
    #### transform the confounds from train and test
    conf_train = ConfoundPipeline.transform(dataset.confounds.loc[trainSubjs])
    conf_test  = ConfoundPipeline.transform(dataset.confounds.loc[testSubjs])

    ### pipe the confounds into the residualization transformers
    brain_pipe = BrainPipeline()
    beh_pipe = BehPipeline(gaussianize=True)
    ### fit and residualize training
    brain_train_clean = brain_pipe.fit_transform(dataset.brainData.loc[trainSubjs], conf_train)
    beh_train_clean = beh_pipe.fit_transform(dataset.behaviorData.loc[trainSubjs], conf_train)

    ### apply transformation / residualize test data
    brain_test_clean  = brain_pipe.transform(dataset.brainData.loc[testSubjs],  conf_test)
    beh_test_clean  = beh_pipe.transform(dataset.behaviorData.loc[testSubjs],  conf_test)
    return {'BrainTrainClean': brain_train_clean,
            'BrainTestClean':   brain_test_clean,
            'BehTrainClean': beh_train_clean,
            'BehTestClean':   beh_test_clean,
            'train_confounds':conf_train,
            'test_confounds':conf_test,
            'conf_pipeline':ConfoundPipeline,
            'BrainPipeline':brain_pipe,
            'BehaviorPipe':beh_pipe}