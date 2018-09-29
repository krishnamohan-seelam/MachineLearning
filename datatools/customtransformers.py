from scipy.stats import boxcox
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class OrdinalTransformer(TransformerMixin):
    """OrdinalTransformer :To transform panda data frame strings into numerical data
    """

    def __init__(self, col, ordering=None):
        """
        Args:
        col: pandas column to  transformed 
        ordering (list): 
        """
        self.col = col
        self.ordering = ordering

    def transform(self, df):
        """OrdinalTransformer :To transform panda data frame strings into numerical data
        returns transformed dataframe
        Args:
        df: pandas dataframe

        """
        X = df.copy()
        X[self.col] = X[self.col].map(lambda x: self.ordering.index(x))
        return X

    def fit(self, df, y=None):
        return self


class DummyTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        return pd.get_dummies(df, self.cols)

    def fit(self, df, y=None):
        return self


class ImputeTransformer(TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
        self.cols = cols
        self.strategy = strategy

    def transform(self, df):
        X = df.copy()
        impute = Imputer(strategy=self.strategy)
        for col in self.cols:
            X[col] = impute.fit_transform(X[[col]])
        return X

    def fit(self, df, y=None):
        return self


class CategoryTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        X = df.copy()
        for col in self.cols:
            X[col].fillna(X[col].value_counts().index[0], inplace=True)
        return X


class BoxcoxTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        X = df.copy()
        for col in self.cols:
            # boxcox is only applicable for positive
            if X[col].min() > 0:
                bc_transformed, _ = boxcox(X[col])
                X[col] = bc_transformed
        return X

    def fit(self, df, y=None):
        return self


class BinCutterTransformer(TransformerMixin):
    def __init(self, col, bins, labels=False):
        self.col = col
        self.bins = bins
        self.labels = labels

    def tranform(self, df):
        X = df.copy()
        X[self.col] = pd.cut(X[self.col], bins=self.bins, labels=self.labels)
        return X

    def fit(self, df, y=None):
        return self


class LogTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        X = df.copy()
        for col in self.cols:
            X[col] = np.log1p(X[col])
        return X

    def fit(self, df, y=None):
        return self


class MinMaxTransformer(TransformerMixin):
    """
    Transforms features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such that it
     is in the given range on the training set, i.e. between zero and one.
    Parameters:
    cols : list of columns to be transformed
    feature_range : tuple (min, max), default=(0, 1)
    copy : boolean, optional, default True

    """

    def __init__(self, cols=None, feature_range=(0, 1), copy=True):
        self.cols = cols
        self.minmax_sc = MinMaxScaler(feature_range, copy)

    def transform(self, df):
        X = df.copy()
        for col in self.cols:
            X[col] = self.minmax_sc.fit_transform(X[[col]])
        return X

    def fit(self, df, y=None):
        return self


class StdScaleTransformer(TransformerMixin):
    def __init__(self, cols=None, copy=True, with_mean=True, with_std=True):
        self.cols = cols
        self.scaler = StandardScaler(
            copy=copy, with_mean=with_mean, with_std=with_std)

    def transform(self, df, y=None):
        X = df.copy()
        for col in self.cols:
            X[col] = self.scaler.fit_transform(X[[col]])
        return X

    def fit(self, df, y=None):
        return self
