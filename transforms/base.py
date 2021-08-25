from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re

class DropByValues(BaseEstimator, TransformerMixin):
    def __init__(self, cols, vals_to_remove):
        self.cols = cols
        self.vals_to_remove = vals_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        drop_idxs = []
        for col in self.cols:
            if len(self.vals_to_remove[col]) == 1 and self.vals_to_remove[col][0] == 'NaN' or self.vals_to_remove[col][
                0] == 'nan':
                drop = [i for i, val in X[col].iteritems() if np.isnan(val)]
            else:
                drop = [i for i, val in X[col].iteritems() if val in self.vals_to_remove[col]]
            drop_idxs.extend(drop)
        X.drop(drop_idxs, inplace=True)
        X.reset_index(drop=True, inplace=True)
        return X


class MissingValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, val_to_replace, replace_vals):
        self.cols = cols
        self.vals_to_replace = val_to_replace
        self.replace_vals = replace_vals

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.cols:
            miss_val, rep_val = self.vals_to_replace[col], self.replace_vals[col]
            if rep_val == "NaN" or rep_val == "nan":
                rep_val = np.NaN

            X = X.replace(miss_val, rep_val)

        return X


class RegexFilter(BaseEstimator, TransformerMixin):
    def __init__(self, regex_filter, cols):
        self.regex_filter = regex_filter
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        drop_idxs = []
        for col in self.cols:
            drop = [i for i, val in X[col].iteritems() if val is np.NaN or self.regex_filter.match(val) is None]
            drop_idxs.extend(drop)
        X.drop(drop_idxs, inplace=True)
        X.reset_index(drop=True, inplace=True)

        return X


class BaseNumericalFeatureEngineer(BaseEstimator, TransformerMixin):
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

        return X
