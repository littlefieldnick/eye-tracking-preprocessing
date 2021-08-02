from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class MissingValTransformer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, missing_val=-1, replace_val=np.nan):
        self.missing_val = missing_val
        self.replace_val = replace_val

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert -1s in pupil size columns to NaN
        X = X.replace(self.missing_val, self.replace_val)
        return X

class NumericalFeatureEngineer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self):
        pass

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, "time_since_stimulus_appeared"] = np.nan
        subs = pd.unique(X["participant_name"])
        stims = pd.unique(X["presented_stimulus_name"])
        for sub in subs:
            for stim in stims:
                if stim is np.nan:
                    continue

                stim_times = X[(X["presented_stimulus_name"] == stim) & (X["participant_name"] == sub)][
                    "recording_timestamp"]

                if len(stim_times) == 0:
                    continue

                start_timestamp = stim_times.iloc[0]
                X.loc[stim_times.index, "time_since_stimulus_appeared"] = stim_times - start_timestamp

        return X.loc[:, "time_since_stimulus_appeared"]

