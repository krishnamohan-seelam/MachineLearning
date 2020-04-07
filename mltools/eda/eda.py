from scipy.stats import chi2_contingency
from scipy import stats
""" EDA - Helper functions to  aid data analysis"""
import pandas as pd


def null_analysis(dataframe):
    """
    desc: get nulls for each column in counts & percentages
    :param frame: DataFrame on which null analysis needs to perfomed
    :returns: Null Analysis  DataFrame
    :rtype: Pandas DataFrame
    """
    null_count = dataframe.isnull().sum()
    null_count = null_count[null_count != 0]
    # calculate null percentages
    null_percent = null_count / len(dataframe) * 100
    null_table = pd.concat(
        [pd.DataFrame(null_count), pd.DataFrame(null_percent)], axis=1)
    null_table.columns = ['counts', 'percentage']
    null_table.sort_values('counts', ascending=False, inplace=True)
    return null_table


def anova(frame, categorical_features, target):
    """
    Calculates Anova  for categorical features
    desc: calculates anova for  categorical features
    :param frame: DataFrame on which anova needs to calculated
    :type frame: Pandas DataFrame
    :param categorical_features:  Categorical Features 
    :type print_cols: List
    :returns: Anova DataFrame
    :rtype: Pandas DataFrame
    """
    anv = pd.DataFrame()
    anv['features'] = categorical_features
    pvals = []
    for c in categorical_features:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls][target].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(colX)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(
                colX)
        print(result)

    def test_chi2(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(
            expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)
