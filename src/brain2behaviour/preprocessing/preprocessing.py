import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from .utils import cube_root,zscore,normal_eqn_python,palm_inormal
from .transformers import * 

def clean_beh_data(dataset,fold,train_confounds,test_confounds,gaussianize=True):
    Y_beh_train=dataset.behaviorData.loc[dataset.cv_folds[fold]['training']]
    Y_beh_test=dataset.behaviorData.loc[dataset.cv_folds[fold]['testing']]

    ## get columns to output as df
    train_cols=Y_beh_train.columns
    test_cols=Y_beh_test.columns

    #### clean behavioral data
    if gaussianize==True:
        #1. gaussianize it 
        rint=PalmInormalTransformer(method='blom',overwrite=True)#fit it
        ## transform it to a gaussian
        Y_beh_train=rint.fit_transform(Y_beh_train)
        Y_beh_test=rint.transform(Y_beh_test)
        
    #2 fit linear model -- pas cleaned confound matrices
    betas=normal_eqn_fit(train_confounds,Y_beh_train)## fit it -- get betas
    
    Y_train_resid=normal_eqn_resid(train_confounds,Y_beh_train,betas)
    Y_test_resid=normal_eqn_resid(test_confounds,Y_beh_test,betas)

    sclr=StandardScaler()
    sclr.fit(Y_train_resid)
    Y_train_resid=sclr.transform(Y_train_resid)
    Y_test_resid=sclr.transform(Y_test_resid)

    Y_train_resid=pd.DataFrame(Y_train_resid,columns=train_cols,index=dataset.cv_folds[fold]['training'])
    Y_test_resid=pd.DataFrame(Y_test_resid,columns=test_cols,index=dataset.cv_folds[fold]['testing'])


    return Y_train_resid,Y_test_resid

class GlobalStandardScaler(BaseEstimator, TransformerMixin):
    """
    Subtracts per-column mean, then divides by *global std of centered data*.
    used to normalize brain data .
    """
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.means_ = np.nanmean(X, axis=0)
        centered = X - self.means_
        self.global_std_ = np.nanstd(centered.flatten())
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        centered = X - self.means_
        return centered / self.global_std_


def clean_brain_data(dataset,fold,train_confounds,test_confounds):
    Y_brain_train=dataset.brainData.loc[dataset.cv_folds[fold]['training']]
    Y_brain_test=dataset.brainData.loc[dataset.cv_folds[fold]['testing']]
    ### get columns so we can output dfs
    train_cols=Y_brain_train.columns
    test_cols=Y_brain_test.columns

    
    #1 we don't gaussianize the brain data but we do standard scale 
    sclr=GlobalStandardScaler()
    sclr.fit(Y_brain_train)
    Y_brain_train=sclr.transform(Y_brain_train)
    Y_brain_test=sclr.transform(Y_brain_test)



    # #2 fit linear model -- pas cleaned confound matrices
    betas=normal_eqn_fit(train_confounds,Y_brain_train)## fit it -- get betas
    
    Y_train_resid=normal_eqn_resid(train_confounds,Y_brain_train,betas)
    Y_test_resid=normal_eqn_resid(test_confounds,Y_brain_test,betas)

    Y_train_resid=pd.DataFrame(Y_train_resid,index=dataset.cv_folds[fold]['training'],columns=train_cols)
    Y_test_resid=pd.DataFrame(Y_test_resid,index=dataset.cv_folds[fold]['testing'],columns=test_cols)

  
    return Y_train_resid,Y_test_resid


def clean_fold(dataset,fold,encode_cols,area_cols,volume_cols,bin_encode=False,passthrough_cols=None,
               gaussianize = True,add_squares = True,zscore_cols=True,drop_cols=None):
    """Clean a single fold of data. Uses SKlearn pipelines -- see transformers.py for more details"""
    print(f'Cleaning {fold}')
    trainSubjs=dataset.cv_folds[fold]['training']
    testSubjs=dataset.cv_folds[fold]['testing']
    if passthrough_cols==None:
        passthrough=[]
    else:
        print(f'{passthrough_cols} are being passed through as is')
        passthrough=passthrough_cols
    
    num_cols = [c for c in dataset.confounds.columns if c not in encode_cols]

    
    #### confound prep logic
    steps = [
        ("label_enc", MultiColumnLabelEncoder(columns=encode_cols)),
        ("cbrt", CubeRootTransformer(columns=volume_cols)),
        ("sqrt", SquareRootTransformer(columns=area_cols)),]
    if bin_encode:
        for key in bin_encode.keys():
            steps.append((f"binarize_{key}", BinarizeColumnTransformer(column=[key], threshold=bin_encode[key])))
        encode_cols=list(set(list(bin_encode.keys())+list(encode_cols))) ### ensures that binary encoded variables are not further transformed
    if add_squares:
        steps.append(("square", SquareTransformer(exclude_cols=encode_cols+passthrough)))
    if gaussianize:
        steps.append(("inormal", PalmInormalTransformer(
        exclude_cols=encode_cols+passthrough,
        method="Blom",overwrite=True)))
    if zscore_cols:
        steps.append(("zscore", StandardScalerDF(columns=None, exclude_cols=encode_cols+passthrough, overwrite=True)))
    
    ConfoundPipeline = Pipeline(steps)

    ConfoundPipeline.fit(dataset.confounds.loc[trainSubjs])
    train_confs=ConfoundPipeline.transform(dataset.confounds.loc[trainSubjs])
    test_confs=ConfoundPipeline.transform(dataset.confounds.loc[testSubjs])


    beh_cleanTrain,beh_cleanTest=clean_beh_data(dataset,fold,train_confs,test_confs)
    brain_cleanTrain,brain_cleanTest=clean_brain_data(dataset,fold,train_confs,test_confs)


    return {'BrainTrainClean': brain_cleanTrain,
            'BrainTestClean':   brain_cleanTest,
            'BehTrainClean': beh_cleanTrain,
            'BehTestClean':   beh_cleanTest}