import os
import numpy as np
import pandas as pd
import pickle
from distpredict.preprocessing.utils import create_cv_groups
from sklearn.model_selection import GroupShuffleSplit ### enables splitting but keeps families together
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from typing import Optional
#### in future iterations, would be good to check for gaussianity before running inverse normal transforms. 
### why transform if not necessary?

class distpredictDataset:
    def __init__(self, brainData, behaviorData, confounds,
                 ncv_splits, filepath: str, subjectList=False, cv_seeds=False,feature_pthresh=0.01) -> None:
        """Dataset class for matrix-based predictive modeling"""
        self.filepath = filepath
        if not os.path.isfile(filepath):
            print(f'New unsaved dataset. Dataset will be saved to {filepath} when .save is called')
        self.subjectList = subjectList
        
        #### initialize empty feature an hyper parameter attributes
        #### these get filled after each step is run externally on cluster. 
        self.features=None
        self.feature_pthresh=feature_pthresh
        self.hyperparameters=None
        self.model_permutation_sets=None

        # Load or assign brain data
        if isinstance(brainData, str):
            if brainData.split('.')[-1]=='csv':
                self.brainData = pd.read_csv(brainData, index_col=0)
            else:
                df=pd.read_parquet(brainData)
                df.index=[int(i) for i in list(df.index)]
                self.brainData = df
                
        else:
            self.brainData = brainData

        # Load or assign behavior data
        if isinstance(behaviorData, str):
            self.behaviorData = pd.read_csv(behaviorData, index_col=0)
        else:
            self.behaviorData = behaviorData

        # Load or assign confounds
        if isinstance(confounds, str):
            self.confounds = pd.read_csv(confounds, index_col=0)
        else:
            self.confounds = confounds

        # Subset by subject list if provided
        if subjectList:
            
            if isinstance(subjectList,str):
                self.subjectList=list(pd.read_csv(subjectList,index_col=0).index)
            else:
                self.subjectList=subjectList
            subs = [int(i) for i in self.subjectList]
            self.brainData = self.brainData.loc[subs]
            self.behaviorData = self.behaviorData.loc[subs]
            self.confounds = self.confounds.loc[subs]

        # Store CV parameters
        self.ncv_splits = ncv_splits
        self.cv_seeds = cv_seeds


        # Array of all subject indices
        self.indices = np.array(self.brainData.index)

        # Consistency checks
        assert len(self.brainData) == len(self.behaviorData), \
            "brain and behavior must match in subject count."
        assert len(self.brainData) == len(self.confounds), \
            "brain and confounds must match in subject count."

        # Determine number of outer/inner splits
        if len(self.ncv_splits) > 1:
            self.n_outer, self.n_inner = self.ncv_splits
        else:
            self.n_outer, self.n_inner = self.ncv_splits[0], None

        # Build reproducible seeds
        if not self.cv_seeds:
            np.random.seed(42)
            seeds = np.random.randint(1, 10000, self.n_outer)
            seeds[0] = 0
            self.cv_seeds = seeds
        else:
            assert len(self.cv_seeds) == self.n_outer, \
                "if providing custom seeds, length must match outer splits."
            self.cv_seeds = np.array(self.cv_seeds)

    # Column management
    def dropConfounds(self, vars2drop):
        self.confounds = self.confounds.drop(vars2drop, axis=1)

    def dropBehaviors(self, vars2drop):
        self.behaviorData = self.behaviorData.drop(vars2drop, axis=1)

    def keepConfounds(self, vars2keep):
        self.confounds = self.confounds[vars2keep]

    def keepBehaviors(self, vars2keep):
        self.behaviorData = self.behaviorData[vars2keep]

    def set_feature(self,featurefilepath):
        if self.features is not None:
            raise Exception("Features already defined")
    def set_hyperparameters(self, hyperparameter_file_path):
        if self.hyperparameters is not None:
            raise Exception("Hyperparameters already defined")
    def set_model_permutation_sets(self, model_permutation_sets_path):
        if self.model_permutation_sets is not None:
            raise Exception("Model permutation sets already defined")

    ######### The following three methods are for splitting when family information is available. 
    # Family-aware K-fold
    #### note this is legacy implementation to ensure that we replicate preprint results
    def gen_CV_FamilyFoldsImplicit(self,familyData,column_name):
        """
        Implicit family-aware K-fold. To replicate the paper, use seeds [0,42,19,123,10,69,33,1,1234,9245].
        """
        self.split_type='family_legacy'

        if isinstance(familyData, str):
            self.familyData = pd.read_csv(familyData, index_col=0)
        else:
            self.familyData = familyData
        
        DataGroups, FamilyGroups = [], []

        fam_ids = None
        for seed in self.cv_seeds:
            np.random.seed(seed)
            if seed == 0:
                print('first iteration use data as is')
                fam_ids = np.unique(self.familyData[column_name])
            else:
                np.random.shuffle(fam_ids)

            family_groups = np.array_split(fam_ids, self.n_inner)
            grouped_data = self.familyData.groupby(column_name)
            DataGroups.append(grouped_data)
            FamilyGroups.append(family_groups)

        # Build per-seed fold sets
        fold_sets = {}
        for idx, (fam_list, grp_data) in enumerate(zip(FamilyGroups, DataGroups), start=1):
            fold_sets[f'fold set {idx}'] = create_cv_groups(fam_list, grp_data)

        # Flatten to cv_folds
        all_folds = {}
        fold_number = 1
        for subdict in fold_sets.values():
            for fold_data in subdict.values():
                all_folds[f'fold{str(fold_number).zfill(3)}'] = fold_data
                fold_number += 1

        self.cv_folds = all_folds

    # family-aware K-fold splits with sklearn 
    def gen_CV_FamilyFoldsSKlearn(self,familyData,frac,nsplits):
        self.split_type='family'

        """
        sklearn group fractional splitting
        """
        if isinstance(familyData, str):
            self.familyData = pd.read_csv(familyData, index_col=0)
        else:
            self.familyData = familyData
        
        DataGroups, FamilyGroups = [], []

        splitter=GroupShuffleSplit(n_splits=nsplits,test_size=frac,random_state=42)
        id_col=self.familyData.columns[0]
        split=splitter.split(self.familyData,groups=self.familyData[id_col])

        all_folds={}
        fold_number = 1
        for train_inds, test_inds in split:
            train = self.familyData.index[train_inds]
            test = self.familyData.index[test_inds]
            all_folds[f'fold{str(fold_number).zfill(3)}']={'training':list(train),'testing':list(test)}
            fold_number+=1
        self.cv_folds = all_folds

        
    def gen_hyperparameterFamilyTuningFolds(self,frac,nsplits):
        id_col=self.familyData.columns[0]
        if not hasattr(self, 'cv_folds'):
            raise ValueError("to subsplit you must have previously generated cv_folds with this dataset.")
        else:
            splitter=GroupShuffleSplit(n_splits=nsplits,test_size=frac,random_state=42)
            
            self.hyper_parameterSplits={}
            
            for fold in self.cv_folds:
                indices=self.cv_folds[fold]['training']
                inner_famData=self.familyData.loc[indices]
                split=splitter.split(inner_famData,groups=inner_famData[id_col])
                x=1
                for train_inds, test_inds in split:
                    train = inner_famData.index[train_inds]
                    test = inner_famData.index[test_inds]
                    self.hyper_parameterSplits[f'{fold}_{x}']={'training':train,'testing':test}
                    x+=1
    #################################
    # get CV splits when no family data is available. 
    #################################

    def gen_CV_FoldsNaive(self,frac,nsplits, random_state=42):
        self.split_type='naive'
        splitter = ShuffleSplit(n_splits=nsplits, test_size=frac, random_state=random_state)
        all_folds = {}
        fold_number = 1
        for train_inds, test_inds in splitter.split(self.indices):
            train = self.indices[train_inds]
            test = self.indices[test_inds]
            all_folds[f'fold{str(fold_number).zfill(3)}'] = {
                'training': list(train),
                'testing': list(test)
            }
            fold_number += 1
    
        self.cv_folds = all_folds

    def gen_hyperparameterTuningFoldsNaive(self,frac,nsplits):
        id_col=self.familyData.columns[0]
        if not hasattr(self, 'cv_folds'):
            raise ValueError("to subsplit you must have previously generated cv_folds with this dataset.")
        else:
            splitter=ShuffleSplit(n_splits=nsplits,test_size=frac,random_state=42)
            
            self.hyper_parameterSplits={}
            for fold in self.cv_folds:
                split=splitter.split(self.cv_folds[fold]['training'])
                x=1
                for train_inds, test_inds in split:
                    train = self.indices[train_inds]
                    test = self.indices[test_inds]
                    self.hyper_parameterSplits[f'{fold}_{x}']={'training':train,'testing':test}
                    x+=1

    ######################################
    # Pickle save/load
    def save(self, filepath: Optional[str] = None,overwrite: bool = False):
        """Serialize this instance to a pickle file."""
        if os.path.isfile(self.filepath) and filepath is None:
            if not overwrite:    
                raise Exception(f"{self.filepath} already exists and no alternative filepath was given")
            else:
                print(f"Warning: Overwriting existing file at {self.filepath}.")
                with open(self.filepath, 'wb') as f:
                    pickle.dump(self, f)
        elif not os.path.isfile(self.filepath):
            with open(self.filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)


    @classmethod
    def load(cls,filepath: str):
        """Load a distpredictDataset instance from a pickle file."""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, distpredictDataset):
            raise TypeError(f"Expected distpredictDataset, got {type(obj)}")
        return obj
