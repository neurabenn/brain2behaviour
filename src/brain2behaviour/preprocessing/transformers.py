import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.special import erfinv
from brain2behaviour.preprocessing.utils import cube_root,zscore,normal_eqn_python,palm_inormal

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, overwrite=True):
        """
        Parameters
        ----------
        columns : list[str]
            Columns to label-encode.
        overwrite : bool
            If True, replace the original columns. If False, add new ones with '_le' suffix.
        """
        self.columns = columns
        self.overwrite = overwrite

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.columns is None:
            self.columns = X.columns.tolist()

        self.encoders_ = {}
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))  # force string to handle mixed types
            self.encoders_[col] = le
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        X_out = X.copy()
        for col in self.columns:
            le = self.encoders_[col]
            encoded = le.transform(X_out[col].astype(str))
            if self.overwrite:
                X_out[col] = encoded
            else:
                X_out[f"{col}_le"] = encoded
        return X_out


class BinarizeColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, threshold=None, positive_label=1, negative_label=0):
        """
        Parameters
        ----------
        column : str
            The column to binarize.
        threshold : float, optional
            If provided, values > threshold get positive_label, else negative_label.
            If None, assumes the column already has exactly two unique values.
        positive_label : int or str
            Value to use for the "positive" class (default=1).
        negative_label : int or str
            Value to use for the "negative" class (default=0).
        """
        self.column = column
        self.threshold = threshold
        self.positive_label = positive_label
        self.negative_label = negative_label

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BinaryBinarizer only works on pandas DataFrames")

        if self.threshold is None:
            unique_vals = pd.Series(X[self.column].unique()).dropna()
            if len(unique_vals) != 2:
                raise ValueError(
                    f"Column {self.column} does not have exactly 2 unique values. "
                    f"Found {list(unique_vals)}"
                )
            self.mapping_ = {
                unique_vals.iloc[0]: self.negative_label,
                unique_vals.iloc[1]: self.positive_label,
            }
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        if self.threshold is not None:
            X_out[self.column] = np.where(
                X_out[self.column] > self.threshold, self.positive_label, self.negative_label
            )
        else:
            X_out[self.column] = X_out[self.column].map(self.mapping_)
        return X_out


class SquareRootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        Parameters
        ----------
        columns : list of str
            Which columns to square.
        """
        self.columns = columns or []

    def fit(self, X, y=None):
        # Nothing to learn for squaring, but we store column names
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.columns_ = self.columns
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.columns_:
            X_out[col] = np.sqrt(X_out[col])
        return X_out

class SquareTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, exclude_cols=None, overwrite=False):
        """
        Parameters
        ----------
        columns : list of str, optional
            Explicit list of columns to square. If None, all columns except those
            in exclude_cols will be squared.
        exclude_cols : list of str, optional
            Columns to exclude from squaring (only used if columns is None).
        overwrite : bool, default=False
            If True, overwrite original column. If False, add new column with suffix '_sq'.
   
        """
        self.columns = columns
        self.exclude_cols = exclude_cols or []
        self.overwrite = overwrite

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if self.columns is None:
            # Use all numeric columns except exclude_cols
            self.columns_ = [c for c in X.columns if c not in self.exclude_cols]
        else:
            self.columns_ = self.columns
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.columns_:
            if self.overwrite:
                X_out[col] = X_out[col] ** 2
            else:
                X_out[f"{col}_sq"] = X_out[col] ** 2
        return X_out
        
class CubeRootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        Parameters
        ----------
        columns : list of str
            Which columns to square.
        """
        self.columns = columns or []

    def fit(self, X, y=None):
        # Nothing to learn for squaring, but we store column names
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.columns_ = self.columns
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.columns_:
            X_out[f"{col}"] = cube_root(X_out[col])
        return X_out

class StandardScalerDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, exclude_cols=None, overwrite=True, ddof=0):
        self.columns = columns
        self.exclude_cols = exclude_cols or []
        self.overwrite = overwrite
        self.ddof = ddof

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.columns is None:
            cols = X.select_dtypes(include=["number"]).columns.difference(self.exclude_cols).tolist()
        elif isinstance(self.columns, str):
            cols = [self.columns]
        else:
            cols = list(self.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        self.columns_ = cols
        self.means_ = X[self.columns_].mean(axis=0).to_dict()
        self.stds_  = X[self.columns_].std(axis=0, ddof=self.ddof).replace({0.0: 1.0}).to_dict()
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.columns_:
            mu, sd = self.means_[col], self.stds_[col]
            X_out[col] = (pd.to_numeric(X_out[col], errors="coerce") - mu) / sd
        return X_out


class PalmInormalTransformer(BaseEstimator, TransformerMixin):
    """
    Learn-and-apply PALM inverse normal using the user's palm_inormal.

    - fit(): calls palm_inormal on TRAIN for each column to get y_train, then
             stores a mapping from each unique original value x -> mean(y_train at x)
             (matches PALM's tie handling exactly: averaging AFTER probit).
    - transform(): uses that learned (x -> y) mapping; exact matches return
                   the same y as in training; other values use linear interpolation.
    - Preserves DataFrame index and column order; untouched cols are unchanged.
    """

    def __init__(self, columns=None, method="Blom", c=None, quanti=False,
                 overwrite=True, clip_eps=1e-7, exclude_cols=None):
        """
        columns : list[str] | str | None
            Columns to transform. If None, all numeric columns are used.
        method, c, quanti : forwarded to palm_inormal (must match your usage).
        overwrite : bool
            If True overwrite the columns; else write '<col>_gauss'.
        clip_eps : float
            Safety clip for interpolation endpoints when needed.
        """
        self.columns   = columns
        self.method    = method
        self.c         = c
        self.quanti    = quanti
        self.overwrite = overwrite
        self.clip_eps  = clip_eps
        self.exclude_cols = exclude_cols or []

    def _ensure_columns(self, X: pd.DataFrame):
        if self.columns is None:
            # auto: all numeric minus excludes
            cols = X.select_dtypes(include=["number"]).columns.difference(self.exclude_cols).tolist()
        elif isinstance(self.columns, str):
            cols = [self.columns]
        else:
            cols = list(self.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")
        return cols

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.columns_ = self._ensure_columns(X)
        self.maps_ = {}

        for col in self.columns_:
            s = pd.to_numeric(X[col], errors="coerce").to_numpy()
            mask = ~np.isnan(s)
            x = s[mask]
            if x.size == 0:
                # Degenerate: learn a trivial map
                self.maps_[col] = {
                    "x_unique": np.array([0.0]),
                    "y_unique": np.array([0.0]),
                    "y_low": 0.0,
                    "y_high": 0.0,
                }
                continue

            # 1) Compute y exactly with theipalm from palm function (ties averaged AFTER probit)
            y_train = palm_inormal(
                x, c=self.c, method=self.method, quanti=self.quanti
            )

            # 2) Build mapping: unique x -> mean(y) for that x (matches PALM tie handling)
            df_xy = pd.DataFrame({"x": x, "y": y_train})
            gp = df_xy.groupby("x", sort=True, as_index=True)["y"].mean()
            x_unique = gp.index.to_numpy()
            y_unique = gp.to_numpy()

            # 3) Endpoint y for values outside train support
            #    (use min/max y_unique; equivalent to clipping to extremes)
            y_low  = float(y_unique[0])
            y_high = float(y_unique[-1])

            self.maps_[col] = {
                "x_unique": x_unique,
                "y_unique": y_unique,
                "y_low": y_low,
                "y_high": y_high,
            }

        return self

    def transform(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        X_out = X.copy()  # preserves index & column order

        for col in self.columns_:
            m = self.maps_[col]
            xu, yu = m["x_unique"], m["y_unique"]
            y_low, y_high = m["y_low"], m["y_high"]

            s = pd.to_numeric(X_out[col], errors="coerce").to_numpy()
            out = np.full_like(s, np.nan, dtype=float)
            mask = ~np.isnan(s)

            # Linear interpolation in (x->y) space; exact matches reproduce training y exactly
            out[mask] = np.interp(s[mask], xu, yu, left=y_low, right=y_high)

            if self.overwrite:
                X_out[col] = out
            else:
                X_out[f"{col}_gauss"] = out

        return X_out

def normal_eqn_python(X,Y):
    X=np.asarray(X,dtype='float32')
    Y=np.asarray(Y,dtype='float32')
    params=np.linalg.pinv(np.dot(X.T,X)).dot(X.T).dot(Y)
    resid=Y-np.dot(params.T,X.T).T
    return resid
    
def normal_eqn_fit(X, Y):
    """
    Fit regression Y ~ X (no intercept).
    Returns parameter matrix beta.
    """
    X = np.asarray(X, dtype="float32")
    Y = np.asarray(Y, dtype="float32")
    beta =np.linalg.pinv(np.dot(X.T,X)).dot(X.T).dot(Y)
    return beta

def normal_eqn_resid(X, Y, beta):
    """
    Compute residuals Y - X @ beta.
    """
    X = np.asarray(X, dtype="float32")
    Y = np.asarray(Y, dtype="float32")
    resid = Y-np.dot(beta.T,X.T).T
    return resid


##### new transformers built for streamlining pipeline
######################################################

class PCASizeDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_components=2, prefix="SizePC", drop_original=True, random_state=0):
        self.cols = list(cols)
        self.n_components = n_components
        self.prefix = prefix
        self.drop_original = drop_original
        self.random_state=random_state
        self.pca_ = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X, y=None):
        self.pca_.fit(X[self.cols])
        print('explained variance in head size by PCA')
        print(self.pca_.explained_variance_ratio_)
        return self

    def transform(self, X, y=None):
        Xo = X.copy()
        scores = self.pca_.transform(Xo[self.cols])
        for i in range(self.n_components):
            Xo[f"{self.prefix}{i+1}"] = scores[:, i]
        if self.drop_original:
            Xo = Xo.drop(columns=self.cols)
        return Xo


class GaussianizeY(BaseEstimator):
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.rint_ = None

    def fit(self, y):
        if isinstance(y, pd.Series): y = y.to_frame()
        if self.enabled:
            self.rint_ = PalmInormalTransformer(method='blom', overwrite=True)
            self.rint_.fit(y)
        return self

    def transform(self, y):
        if isinstance(y, pd.Series): y = y.to_frame()
        return self.rint_.transform(y) if self.enabled and self.rint_ is not None else y.copy()

class ResidualizeY(BaseEstimator):
    def __init__(self):
        self.betas_ = None

    def fit(self, conf, y):
        # y must be a DataFrame to grab index/columns later
        if isinstance(y, pd.Series):
            y = y.to_frame()
        self._train_index = y.index
        self._train_cols  = y.columns
        self.betas_ = normal_eqn_fit(conf, y)
        return self

    def transform(self, conf, y):
        if isinstance(y, pd.Series):
            y = y.to_frame()
        res = normal_eqn_resid(conf, y, self.betas_)  # <- likely ndarray
        # wrap back to DataFrame with y's labels
        return pd.DataFrame(res, index=y.index, columns=y.columns)

class StandardizeY(BaseEstimator):
    def __init__(self):
        self.scaler_ = StandardScaler()

    def fit(self, y):
        self.scaler_.fit(y)
        return self

    def transform(self, y):
        return pd.DataFrame(self.scaler_.transform(y), index=y.index, columns=y.columns)


#### deprecated functoin -- useful at somempoint posisbly?
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
