import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


def auto_scatter_simple(df , plot_cols , target_cols , filename):
    matplotlib.use('agg')
    for target_col in target_cols:

        for col in plot_cols:
            fig = plt.figure(figsize=(6 , 6))
            ax = fig.gca()
            ## simple scatter plot
            df.plot(kind='scatter' , x=col , y=target_col , ax=ax ,
                    color='DarkBlue')
            ax.set_title('Scatter plot of {0} vs. {1}'.format(target_col , col))
            outputfile = getoutFileName(filename ,target_col, 'png')
            print(outputfile)
            fig.savefig(outputfile)
        return plot_cols


def getoutFileName(inputfile , target_col , extension):
    fileName , fileExtension = os.path.splitext(inputfile)
    print(fileName , fileExtension)
    extension = "." + extension

    if not fileExtension:
        return fileName + target_col + extension

    if not (fileExtension and not fileExtension.isspace()):
        return fileName + target_col + extension

    if not inputfile.endswith(extension):
        return inputfile.replace(fileExtension , target_col + extension)


def load_data(filename , separator=",") -> pd.DataFrame:
    data_set = None
    if not os.path.exists(filename):
        print("{0} file not found".format(filename))
        raise ValueError("{0} not found".format(filename))

    if filename:
        try:
            data_set = pd.read_csv(filename , sep=separator , header=0 ,
                                   encoding='utf8')
        except FileNotFoundError:
            print("{0} file not found".format(filename))
            return None
    return data_set


def load_dataset(input_file , response , colseparator=','):
    input_dataset = load_data(input_file , colseparator)
    print(" input file is :{0} loaded.".format(input_file))
    # print(input_dataset.head())

    try:
        continuous_vars = input_dataset.describe().columns.values.tolist()
        print("Continous Variables")
        print(continuous_vars)
    except ValueError:
        print("No continous variables")

    try:
        categorical_vars = input_dataset.describe(
            include=["object"]).columns.values.tolist()
        print("Categorical Variables")
        print(categorical_vars)
    except ValueError:
        print("No categorical variables")
        categorical_vars = None

    response_column = [col for col in input_dataset.columns if response == col]
    feature_columns = [col for col in input_dataset.columns if response != col]

    return (input_dataset , feature_columns , response_column ,continuous_vars ,
            categorical_vars)


def print_dataset_info(dataset):
    if dataset is None:
        print("data set is EMPTY")
    else:
        print("No of Observation:{0}".format(dataset.shape[0]))
        print("No of features:{0}".format(dataset.shape[1]))
        print("Features:{0}".format(dataset.columns.values))
        print("Describe dataset:{0}".format(dataset.describe()))


def split_dataset(train_X , train_Y , ptest_size=0.3 , prandom_state=1):
    (train_data , test_data , train_target , test_target) = train_test_split(
        train_X , train_Y , test_size=ptest_size , random_state=prandom_state)

    return (train_data , test_data , train_target , test_target)


def detect_outliers(dataset , noutliers , columns):
    outlier_indices = []
    for column in columns:
        # 1st quartile (25%),# 3rd quartile (75%)
        q1 , q3 = np.percentile(dataset[column] , [25 , 75])

        # Interquartile range (IQR)
        iqr = q3 - q1

        # outlier step
        outlier_step = 1.5 * iqr

        lower_bound = q1 - outlier_step
        upper_bound = q3 + outlier_step

        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataset[(dataset[column] < lower_bound) | (
        dataset[column] > upper_bound)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
 
    multiple_outliers = list(k for k , v in outlier_indices.items()
                             if v > noutliers)

    return multiple_outliers


def display_data_descriptives(input_dataset , feature_columns ,
                              response_column):
    print("<{0} {1} {0}>".format("=" * 40 , "info"))
    print(input_dataset.info())
    print("<{0} {1} {0}>".format("=" * 40 , "feature columns"))
    print(feature_columns)
    print("<{0} {1} {0}>".format("=" * 40 , "response"))
    print(response_column)
    print("<{0} {1} {0}>".format("=" * 40 , "Descriptive Statistics -X"))
    print(input_dataset[feature_columns].describe())
    print("<{0} {1} {0}>".format("=" * 40 , "Descriptive Statistics -y"))
    print(input_dataset[response_column].describe())


def print_compare_results(test_target , prediction_dataset):
    test_values = test_target.values.reshape(-1 , 1)
    for i , prediction in enumerate(prediction_dataset.reshape(-1 , 1)):
        print('Predicted: ' , prediction)
        print('Target: ' , test_values[i])

def one_hot_dataframe(data,columns,replace=False):
    fe_vec= feature_extraction.DictVectorizer()
    make_dict = lambda row :dict((column,row[column]) for column in  columns)
    vector_data=pd.DataFrame(fe_vec.fit_transform(
                             data[columns].apply(make_dict, axis=1)).toarray())
    vector_data.columns = fe_vec.get_feature_names()
    vector_data.index= data.index
    if replace:
        data = data.drop(columns, axis=1)
        data = data.join(vector_data)
    return data,vector_data


def plot_learning_curve(train_sizes, train_scores, validation_scores, scoring='neg_mean_squared_error', ylim=(0,50)):
    plt.style.use('seaborn')
    plt.figure()
    plt.title("Learning Curve")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    if scoring == 'neg_mean_squared_error':
        train_scores_mean = -train_scores_mean
        validation_scores_mean = -validation_scores_mean
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(max(-3, validation_scores_mean.min() - .1), train_scores_mean.max() + .1)
    plt.legend(loc="best")
    plt.show()

class DataFrameSelector(BaseEstimator , TransformerMixin):
    def __init__(self , attribute_names):
        self.attribute_names = attribute_names

    def fit(self , X , y=None):
        return self

    def transform(self , X):
        return X[self.attribute_names].values
