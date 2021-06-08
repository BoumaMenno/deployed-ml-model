# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:51:44 2021

@author: Menno
"""
# data manipulation
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# dumping and loading
import joblib
import pickle

# sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# from feature-engine
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder,
)

from feature_engine.transformation import LogTransformer
from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

# load config
with open("config.pickle", "rb") as f:
    config = pickle.load(f)


# import preprocessors as pp

# pre-process data
data = pd.read_csv('data/train.csv')
data['MSSubClass'] = data['MSSubClass'].astype('O')

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), # predictive variables
    data['SalePrice'], # target
    test_size=0.1, # portion of dataset to allocate to test set
    random_state=42, # we are setting the seed here
)

y_train = np.log(y_train)
y_test = np.log(y_test)
