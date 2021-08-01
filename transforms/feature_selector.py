from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features, convert_lower=False, space_to_underscore=False):
        self.features = features
        self.convert_lower = convert_lower
        self.space_to_underscore = space_to_underscore

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return self.features

    def transform(self, X, y=None):
        if self.convert_lower:
            X.columns = [col.lower() for col in X.columns]
            self.features = X.columns
        if self.space_to_underscore:
            X.columns = [col.replace(" ", "_") for col in X.columns]
            self.features = X.columns

        X = X[self.features]
        return X

class BaseFeatureSelector(FeatureSelector):
    def __init__(self, features, stim_filter, inval_mvmnts):
        FeatureSelector.__init__(self, features)
        self.stim_filter = stim_filter
        self.inval_mvmnts = inval_mvmnts

    def _remove_invalid_eye_mvments(self, X):
        drop_idx = [i for i, val in X["eye_movement_type"].iteritems() if val in self.inval_mvmnts]
        X = X.drop(drop_idx)
        return X

    def _filter_stimulus_names(self, X):
        drop_idx = [i for i, val in X["presented_stimulus_name"].iteritems() if val is not np.NaN and self.stim_filter.match(val) is None]
        X = X.drop(drop_idx)
        return X

    def transform(self, X, y=None):
        X = X[self.features]
        X.columns = [col.lower().replace(" ", "_") for col in X.columns]
        self.features = X.columns

        X.loc[:, "eye_movement_type"] = X["eye_movement_type"].astype("category")
        print(X)
        X = self._remove_invalid_eye_mvments(X)
        X = self._filter_stimulus_names(X)

        return X