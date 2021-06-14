# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:26:16 2021

@author: nlmboum
"""
from sklearn.base import BaseEstimator, TransformerMixin


# categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        if not isinstance(mappings, dict):
            raise ValueError('mappings should be a dictionary')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(self.mappings)

        return X