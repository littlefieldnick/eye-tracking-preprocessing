from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features, step=None, dropna=False, convert_lower=False, space_to_underscore=False):
        self.features = features
        self.step = step
        self.dropna = dropna
        self.convert_lower = convert_lower
        self.space_to_underscore = space_to_underscore

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return self.features

    def transform(self, X, y=None):
        if self.convert_lower:
            self.features = [col.lower() for col in self.features]

        if self.space_to_underscore:
            self.features = [col.replace(" ", "_") for col in self.features]

        X = X[self.features]
        
        mask_na = [i for i, val in X["presented_stimulus_name"].iteritems() if val is np.NaN]
                
        X = X.drop(mask_na)
        
        if self.dropna and self.step == "pupil":
            nas_left = [i for i, val in X["pupil_diameter_left"].iteritems() if np.isnan(val)]
            nas_right = [i for i, val in X["pupil_diameter_right"].iteritems() if np.isnan(val)]

            X = X.drop(nas_left + nas_right)

        if self.step == "aoi":
            mask = (X["aoi_hit_[box:bottom]"] == 1) & (X["aoi_hit_[box:top]"] == 1)
            X = X.loc[-mask, :]
            
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
        drop_idx = [i for i, val in X["presented_stimulus_name"].iteritems() if val is np.NaN or self.stim_filter.match(val) is None]
        X = X.drop(drop_idx)
        return X

    def transform(self, X, y=None):
        X = X[self.features]
        X.columns = [col.lower().replace(" ", "_") for col in X.columns]
        self.features = X.columns

        X.loc[:, "eye_movement_type"] = X["eye_movement_type"].astype("category")
        X = self._remove_invalid_eye_mvments(X)
        X = self._filter_stimulus_names(X)

        return X
