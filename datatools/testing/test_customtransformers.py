import unittest
import numpy as np
import pandas as pd
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from customtransformers import OrdinalTransformer
from pandas.util.testing import assert_frame_equal,assert_series_equal
print(SCRIPT_DIR)


class TestOrdinalTransformer(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({'city': ['tokyo', None, 'london', 'seattle',
                                        'san francisco', 'tokyo'],
                               'boolean': ['yes', 'no', None, 'no', 'no', 'yes'],
                               'ordinal_column': ['somewhat like', 'like',
                                                  'somewhat like', 'like', 'somewhat like',
                                                  'dislike'],
                               'quantitative_column': [1, 11, -.5, 10,
                                                       None, 20]})

    def test_ordinal(self):
        ot = OrdinalTransformer(col='ordinal_column',
                                ordering=['dislike', 'somewhat like', 'like'])
        self.X = ot.fit_transform(self.X)
        test = pd.DataFrame({'ordinal_column': [1, 2, 1, 2, 1, 0]})
        assert_series_equal(self.X['ordinal_column'], test['ordinal_column'])


if __name__ == '__main__':
    unittest.main()
