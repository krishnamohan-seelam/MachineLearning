
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import logging
import sys
from mltools.modelbuilder.basemodel import BaseLoader
from mltools.modelbuilder.dfloader import DataFrameLoader

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              datefmt='%d-%b-%y %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
class SupervisedDataLoader(BaseLoader):

    def __init__(self, train_file, test_file, response):
        self.train_file = train_file
        self.test_file = test_file
        self.response = response
        self._train_df = None
        self._test_df = None
    @property
    def train_dataset(self):
        if self._train_df is None:
            raise ValueError("Training Dataset Not Found")
        return self._train_df

    @property
    def test_dataset(self):
        if self._test_df is None:
            raise ValueError("Test Dataset Not Found")
        return self._testdf

    @property
    def response(self):
        return self._response
    
    @response.setter
    def response(self,value):
        
        if value is None:
            raise ValueError("Supervised Model must have response variable")
        self._response =value
        
    def load(self,*args ,**kwargs):
        """
        Loads datasets based on input train file ,test file and returns dataframes
        Parameters:
        -----------
        args: training dataset's file name
        kwargs : test dataset's file name
        
        """
        logger.info("Loading train_file :{0}".format(self.train_file))
        self._train_df = DataFrameLoader(self.train_file , *args ,**kwargs).dataframe
        logger.info("Loading test_file :{0}".format(self.test_file))
        self._testdf = DataFrameLoader(self.test_file, *args ,**kwargs).dataframe
        return self._train_df, self._testdf

    def describe_target(self):
        if self.response and self.response in self._train_df.columns.values:
            return self._train_df[self.response].describe()

    def get_target_plot(self, continuous=True, convert_log=False):
        fig = plt.figure(figsize=(6, 4))
        #style = dict(size=10, color='gray')
        if continuous:
            ax1 = fig.gca()
            response = np.log1p(
                self._train_df[self.response]) if convert_log else self._train_df[self.response]
            sns.distplot(response, ax=ax1, hist_kws=dict(alpha=1))
            # print("skewness of response: {0}".format(response.skew()))
            # print("kurtosis of response: {0}".format(response.kurtosis()))
        return plt

    def get_feature_groups(self, dataset =None):
        numeric_features = category_features = []
        dataset = self._train_df if dataset is None  else dataset 
        numeric_features = dataset.select_dtypes(include=[np.number]).columns
        category_features = dataset.select_dtypes(include=['object']).columns
        return list(numeric_features), list(category_features)

    def convert_to_str(self, features, to='str'):

        if isinstance(features, collections.Iterable):
            features = list(features)
        for feature in features:
            self._train_df[feature] = self._train_df[feature].astype(to)
            self._testdf[feature] = self._testdf[feature].astype(to)
