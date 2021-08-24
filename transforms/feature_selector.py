from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return self.features

    def transform(self, X, y=None):
        return X[self.features]
