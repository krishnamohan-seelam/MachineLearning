from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


class DataSetPreprocessor(object):

    def __init__(self, train_df, test_df):

        self.train_df = train_df
        self.test_df = test_df
        self._steps = []
        self._pipeline = None

    def add_step(self, func, inverse_func=None, validate=False, kw_args=None):
        """Add step"""
        next_ft = FunctionTransformer(func,
                                      inverse_func=inverse_func,
                                      validate=validate,
                                      kw_args=kw_args,
                                      )
        self._steps.append((func.__name__, next_ft))

    def add_steps(self, funcs):
        for func in funcs:
            f, invf, validate, kw_args = func
            self.add_step(f, invf, validate, kw_args)

    def fit_steps(self):
        """ Fits all steps """
        if not self._pipeline:
            self._pipeline = Pipeline(self._steps)

        self.train_df = self._pipeline.fit_transform(self.train_df)
        self.test_df = self._pipeline.transform(self.test_df)
        return self.train_df, self.test_df
