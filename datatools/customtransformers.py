from scipy.stats import boxcox
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

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

    def fit(self, *_):
        return self


class DummyTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        return pd.get_dummies(df, self.cols)

    def fit(self, *_):
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

    def fit(self, *_):
        return self


class CategoryTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, *_):
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

    def fit(self, *_):
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

    def fit(self, *_):
        return self


class LogTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        X = df.copy()
        for col in self.cols:
            X[col] = np.log1p(X[col])
        return X

    def fit(self, *_):
        return self
