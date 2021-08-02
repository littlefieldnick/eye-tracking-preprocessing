from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class PupilOperation(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, op):
        self.op = op

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    def _diff(self, X):
        return X.diff()

    def _avg(self, X):
        avg = 0
        for col in X.columns:
            avg += X.loc[:, col]
        return avg / len(X.columns)

    def transform(self, X, y=None):
        if self.op == "difference":
            return self._diff(X)
        elif self.op == "average":
            return self._avg(X)

        return X

class PupilSignificance(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, thresh):
        self.threshold = thresh

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    def _diff(self, X):
        return X.diff()

    def _avg(self, X):
        avg = 0
        for col in X.columns:
            avg += X.loc[:, col]
        return avg / len(X.columns)

    def transform(self, X, y=None):
        X = self._avg(X)
        X = self._diff(X)

        signif_change = X > self.threshold
        return np.array([1 if signif else 0 for signif in signif_change])


