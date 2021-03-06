"""
Inspired from Randal S. Olson's autoclean.
Objective of this class to split dataframe into categorical & continuous frames. 
"""
from __future__ import print_function
import numpy as np
import pandas as pd

from os.path import splitext
VALID_FORMATS = ["csv"]


class DataFrameLoader():
    """
    Objective of this class to split dataframe into categorical & continuous frames.
    Properties:
    dataframe
    categorical_features
    continuous_features
    continuous_dataframe
    categorical_dataframe
    """
    loaders   = { 'csv': 'load_csv'}
    def __init__(self, *args, **kwargs):
        self.__dataframe = self.load_data(*args, **kwargs)
        self.__categorical_features = []
        self.__continuous_features = []
        if not self.__dataframe.empty:
            self.__categorical_features = self._get_features_by_type(
                type="object")
            self.__continuous_features = self._get_features_by_type(
                type=np.number)

    @property
    def dataframe(self):
        return self.__dataframe
     
    @property
    def categorical_features(self):
        """ returns categorical features of dataframe as popo list.
            popo - plain old python object
        """
        return list(self.__categorical_features)

    @property
    def continuous_features(self):
        """ 
        returns continuous features of dataframe as popo list.
        """
        return list(self.__continuous_features)

    @property
    def continuous_dataframe(self):
        """ returns dataframe based on continuous_features.
        """
        return self.__dataframe[self.continuous_features]

    @property
    def categorical_dataframe(self):
        """ returns dataframe based on categorical_features.
        """
        return self.__dataframe[self.categorical_features]

    def load_data(self, *args, **kwargs):
        """ invokes respective load method based on extension.
        Args:
        *args:
        **kwargs:
        """
        _, extension = splitext(args[0])
        extension  = extension[1:].lower() 
        return self._getloader(extension,*args, **kwargs)

    def _getloader(self, extension,*args,**kwargs):
        loader = self.loaders.get(extension,'load_csv')
        if hasattr(self,loader):
            return getattr(self,loader)(*args,**kwargs)

    def load_csv(self, *args, **kwargs):
        """ returns panda dataframe  using *args, **kwargs .
        Args:
        *args:
        **kwargs:
        """
        dataframe = pd.DataFrame()
        try:
            dataframe = pd.read_csv(*args, **kwargs)
             
        except Exception:
            raise ValueError("Error in loading data, empty data frame returned")

        return dataframe

    def _get_features_by_type(self, type):
        features = self.__dataframe.describe(include=[type]).columns.values
        features = features if features.any() else []
        return features
