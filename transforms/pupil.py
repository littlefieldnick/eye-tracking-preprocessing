from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import Binarizer


class PupilAggregator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.loc[:, "diff_pupil_diameter_left"] = np.round(X.loc[:, "pupil_diameter_left"].diff(), 3)
        X.loc[:, "diff_pupil_diameter_right"] = np.round(X.loc[:, "pupil_diameter_right"].diff(),3)

        X.loc[:, "avg_pupil_diameter"] = np.round((X.loc[:, "pupil_diameter_left"] + X.loc[:, "pupil_diameter_right"]) / 2, 3)
        X.loc[:, "diff_avg_pupil_diameter"] = np.round(X.loc[:, "avg_pupil_diameter"].diff(), 3)

        return X


class BinarizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, col, thresh):
        self.col = col
        self.thresh = thresh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        binarizer = Binarizer(threshold=self.thresh)

        # Convert NaN to -1 for thresholding

        X.loc[0, self.col] = -1
        X.loc[:, "sign_diff_in_pupil_size"] = binarizer.fit_transform([X[self.col]])[0]

        # Convert -1 back to NaN
        X.loc[0, self.col] = np.NaN

        return X