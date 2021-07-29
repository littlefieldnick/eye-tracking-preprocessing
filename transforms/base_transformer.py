from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class BaseTransformer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, stim_regex, inval_eye_mvmnts, time_since=True):
        self._stim_regex = stim_regex
        self._inval_mvmnts = inval_eye_mvmnts
        self._time_since = time_since

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Check if needed
        if self._time_since:
            # create new column
            X['Time Since Stimulus Appeared'] = X["Recording timestamp"].diff()

        # Filter Stimulus Name
        mask = X["Presented Stimulus name"].apply(lambda stim: stim is not np.NaN and self._stim_regex.match(stim) is None)
        X = X.loc[-mask, :]

        # Filter out invalid eye movements
        mask = X["Eye movement type"].apply(lambda mvmnt: mvmnt in self._inval_mvmnts)
        X = X.loc[-mask, :]

        # Convert -1s in pupil size columns to NaN
        X["Pupil diameter left"] = X[["Pupil diameter left"]].replace([-1], np.nan)
        X["Pupil diameter right"] = X[["Pupil diameter right"]].replace([-1], np.nan)

        # returns a numpy array
        return X