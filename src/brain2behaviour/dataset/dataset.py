import os
import numpy as np
import pandas as pd
import pickle
import math
import collections
from brain2behaviour.preprocessing.utils import create_cv_groups
from sklearn.model_selection import GroupShuffleSplit ### enables splitting but keeps families together
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from typing import Optional
#### in future iterations, would be good to check for gaussianity before running inverse normal transforms. 
### why transform if not necessary?

class BrainBehaviorDataset:
    """
    Container that keeps **all data and cross-validation metadata for a single
    brain-to-behaviour prediction experiment** in one place – it can be
    pickled, shipped to the cluster, updated step-by-step, and saved back to
    disk.

    Parameters
    ----------
    brainData : (pd.DataFrame | str)
        NxP matrix of brain features **or** path to a .csv/.parquet file
        holding that matrix (rows = subjects; columns = edges/nodes).
    behaviorData : (pd.DataFrame | str)
        NxQ matrix / file of behavioural targets to predict.
    confounds : (pd.DataFrame | str)
        NxR matrix / file of nuisance regressors to regress out.
    ncv_splits : tuple[int] | list[int]
        `(n_outer, n_inner)` specifying outer CV folds and optional inner
        hyper-parameter folds.  If length==1 only outer CV is used.
    filepath : str
        Location where the pickled dataset will be written/read.
    subjectList : list[int] | str | False, optional
        Optional list (or .csv file with an index column) specifying which
        subject IDs to keep.  If `False` all rows are used.
    cv_seeds : list[int] | False, optional
        Sequence of seeds – one per outer fold – to make CV reproducible.  If
        `False`, seeds are generated and stored automatically.
    feature_pthresh : float, default=0.01
        Default p-value threshold applied by the CPM feature selector.

    Notes
    -----
    * After initialisation:
        - `self.brainData`, `self.behaviorData`, `self.confounds` are always
          in-memory DataFrames cut to the same subjects.
        - `self.features`, `self.hyperparameters`, `self.model_permutation_sets`
          start as `None` and are filled in later pipeline steps.
        - CV fold dictionaries are created by `gen_*` methods and live in
          `self.cv_folds` and `self.hyper_parameterSplits`.
    * The class provides `.save()` / `.load()` helpers so the **exact state** of
      the experiment (selected subjects, generated folds, tuned hyper-params,
      etc.) can be snapshotted and resumed anywhere on the cluster.

    """
    def __init__(self, brainData, behaviorData, confounds,
                 ncv_splits, filepath: str, subjectList=False, cv_seeds=False,feature_pthresh=0.01) -> None:
        
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

    ##### things to add -- a split which strafies based on family data
    #### that is, treat family as a set of labels to predict 
    #### stratify them across splits so they are equally represented.
    ### see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    

    #######################################
    #### collect features
    def collect_features(
        self,
        feature_dir: str,
        threshold_percentage: float = 100.0,
    ):
        """
        For every column (task) in the Parquet files, keep the edges that
        1. appear in ≥ ``threshold_percentage`` % of folds, **and**
        2. have the same sign in every fold where they appear.
    
        Returns
        -------
        dict[str, dict[str, set[str]]]
            {task_name: {"positive": {edge, …}, "negative": {edge, …}}, …}
        """
        # ---------- validation ---------------------------------------------------
        if not 0.0 <= threshold_percentage <= 100.0:
            raise ValueError("threshold_percentage must be between 0 and 100.")
    
        fold_ids = list(self.cv_folds.keys())
        n_folds  = len(fold_ids)
        if n_folds == 0:
            raise ValueError("self.cv_folds is empty — create CV splits first.")
    
        min_fold_count = math.ceil(threshold_percentage * n_folds / 100.0)
    
        # ---------- tallies for every task ---------------------------------------
        presence  = collections.defaultdict(collections.Counter)  # task → Counter(edge → folds_seen)
        pos_seen  = collections.defaultdict(collections.Counter)  # task → Counter(edge → #pos)
        neg_seen  = collections.defaultdict(collections.Counter)  # task → Counter(edge → #neg)
    
        for fold in fold_ids:
            path = os.path.join(feature_dir, f"{fold}.parquet")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing feature file: {path}. Make sure you run have extracted features prior to running")
    
            df = pd.read_parquet(path)                # index = edge, columns = tasks
            for task in df.columns:                   # iterate over each column once
                series = df[task].dropna()            # ignore NaNs (edge absent)
                idx    = series.index
    
                presence[task].update(idx)            # edge seen in this fold
                pos_seen[task].update(idx[series > 0])
                neg_seen[task].update(idx[series < 0])
                # zeros are ignored for sign consistency
    
        # ---------- apply threshold & sign check ---------------------------------
        features = {}
    
        for task in presence:                         # iterate over discovered tasks
            pos_edges, neg_edges = set(), set()
    
            for edge, folds_seen in presence[task].items():
                if folds_seen < min_fold_count:
                    continue                          # fails presence threshold
                if pos_seen[task][edge] and neg_seen[task][edge]:
                    continue                          # sign flips → discard
    
                if pos_seen[task][edge]:
                    pos_edges.add(edge)
                elif neg_seen[task][edge]:
                    neg_edges.add(edge)
                # edges that were only zero are ignored
    
            features[task] = {"positive": pos_edges, "negative": neg_edges}
    
        # ---------- stash & return ----------------------------------------------
        self.features = features

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
        """Load a brain2behviour dataset instance from a pickle file."""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, b2b_Dataset):
            raise TypeError(f"Expected b2b_Dataset, got {type(obj)}")
        return obj
