
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mltools.modelbuilder.basemodel import BaseLoader
from mltools.modelbuilder.dfloader import DataFrameLoader
import collections


class SupervisedDataLoader(BaseLoader):

    def __init__(self, train_file, test_file, response):
        self.__train_df, self.__testdf = self._load_datasets(
            train_file, test_file)
        if response is None:
            raise ValueError("Supervised Model must have response variable")
        self.response = response

    @property
    def train_dataset(self):
        return self.__train_df

    @property
    def test_dataset(self):
        return self.__testdf

    def _load_datasets(self, train_file=None, test_file=None, sep=','):
        """
        Loads datasets based on input train file ,test file and returns dataframes
        Parameters:
        -----------
        train_file: training dataset's file name
        test_file : test dataset's file name
        
        """
        print("Loading train_file :{0}".format(train_file))
        train_dataframe = DataFrameLoader(train_file, sep).dataframe
        print("Loading test_file :{0}".format(test_file))
        test_dataframe = DataFrameLoader(test_file, sep).dataframe
        return train_dataframe, test_dataframe

    def describe_target(self):
        if self.response and self.response in self.__train_df.columns.values:
            return self.__train_df[self.response].describe()

    def get_target_plot(self, continuous=True, convert_log=False):
        fig = plt.figure(figsize=(6, 4))
        if continuous:
            ax1 = fig.gca()
            response = np.log1p(
                self.__train_df[self.response]) if convert_log else self.__train_df[self.response]
            sns.distplot(response, ax=ax1, hist_kws=dict(alpha=1))
            print("skewness of response: {0}".format(response.skew()))
            print("kurtosis of response: {0}".format(response.kurtosis()))
        return plt

    def get_feature_groups(self, dataset =None):
        numeric_features = category_features = []
        dataset = self.__train_df if dataset is None  else dataset 
        numeric_features = dataset.select_dtypes(include=[np.number]).columns
        category_features = dataset.select_dtypes(include=['object']).columns
        return list(numeric_features), list(category_features)

    def convert_to_str(self, features, to='str'):

        if isinstance(features, collections.Iterable):
            features = list(features)
        for feature in features:
            self.__train_df[feature] = self.__train_df[feature].astype(to)
            self.__testdf[feature] = self.__testdf[feature].astype(to)


if __name__ == '__main__':
    sm = SupervisedDataLoader(train_file='train', test_file='test', response="y")
