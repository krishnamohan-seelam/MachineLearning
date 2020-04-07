import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import chi2_contingency


def missing_values(dataframe):
    missing_values = dataframe.isnull().sum()
    missing_percent = missing_values * 100 / len(dataframe)
    missing_val_table = pd.concat([missing_values, missing_percent], axis=1)
    # print(missing_val_table)

    missing_val_table = missing_val_table.rename(
        columns={0: 'Missing_Values', 1: 'Percent_of_Total_Values'})
    missing_val_table.index.names = ['Feature']
    missing_val_table.reset_index(level=0, inplace=True)
    sorted_missing = missing_val_table[missing_val_table['Missing_Values'] != 0].sort_values(
        'Percent_of_Total_Values', ascending=False).round(2)
    return sorted_missing


def detect_outliers(dataset, noutliers, columns):
    outlier_indices = []
    for column in columns:
        # 1st quartile (25%),# 3rd quartile (75%)
        q1, q3 = np.percentile(dataset[column], [25, 75])
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
    multiple_outliers = list(
        k for k, v in outlier_indices.items() if v > noutliers)
    return multiple_outliers
