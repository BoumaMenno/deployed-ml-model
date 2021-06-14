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
import yaml

import transformers

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

from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.transformation import LogTransformer
from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

# load config
with open('config.yml') as infile:
      config = yaml.load(infile, Loader=yaml.FullLoader)


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

X_train[config["FEATURES"]]
X_test[config["FEATURES"]]

y_train = np.log(y_train)
y_test = np.log(y_test)

pipeline = Pipeline([
    # impute categorical variables with string missing
    ('missing_imputation', CategoricalImputer(imputation_method='missing', fill_value='NA', variables=config["CATEGORICAL_WITH_HIGH_MISSING"])),

    ('frequent_imputation', CategoricalImputer(imputation_method='frequent', variables=config["CATEGORICAL_WITH_MISSING"])),

    # add missing indicator
    ('missing_indicator', AddMissingIndicator(variables=config["NUMERICAL_WITH_MISSING"])),

    # impute numerical variables with the mean
    ('mean_imputation', MeanMedianImputer(imputation_method='mean', variables=config["NUMERICAL_WITH_MISSING"])),
    
    # calculate ages from year values
    ('elapsed_time', CombineWithReferenceFeature(variables_to_combine=config["REFERENCE_FEATURE"], reference_variables=config["YEAR_FEATURES"])),

    ('drop_features', DropFeatures(features_to_drop=config["REFERENCE_FEATURE"])),

    # apply mappings to categorical features
    ('mapper_qual', transformers.Mapper(variables=config["QUALITY_FEATURES"], mappings=config["QUALITY_MAP"])),

    ('mapper_exposure', transformers.Mapper(variables=config["BASEMENT_EXPOSURE_FEATURES"], mappings=config["BASEMENT_EXPOSURE_MAP"])),

    ('mapper_finish', transformers.Mapper(variables=config["FINISH_FEATURES"], mappings=config["FINISH_MAP"])),

    ('mapper_garage', transformers.Mapper(variables=config["GARAGE_FINISH_FEATURES"], mappings=config["GARAGE_FINISH_MAP"])),

    ('mapper_fence', transformers.Mapper(variables=config["FENCE_FEATURES"], mappings=config["FENCE_MAP"])),

    # label all rare values as 'rare'
    ('rare_label_encoder', RareLabelEncoder(tol=0.01, n_categories=1, variables=config["UNMAPPED_CATEGORICAL_FEATURES"])),

    # encode categorical and discrete variables using the target mean
    ('categorical_encoder', OrdinalEncoder(encoding_method='ordered', variables=config["UNMAPPED_CATEGORICAL_FEATURES"])),
    
    ('scaler', MinMaxScaler()),
#     ('selector', SelectFromModel(Lasso(alpha=0.001, random_state=0))),
    ('Lasso', Lasso(alpha=0.001, random_state=0)),
])

# X_train_trans = pd.DataFrame(pipeline.fit_transform(X_train, y_train))
# print(X_train_trans.isna().sum().sort_values(ascending=False))

# make predictions for train set
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_train)

# determine mse, rmse and r2
print('train mse: {}'.format(int(
    mean_squared_error(np.exp(y_train), np.exp(pred)))))
print('train rmse: {}'.format(int(
    mean_squared_error(np.exp(y_train), np.exp(pred), squared=False))))
print('train r2: {}'.format(
    r2_score(np.exp(y_train), np.exp(pred))))
print()

# make predictions for test set
pred = pipeline.predict(X_test)

# determine mse, rmse and r2
print('test mse: {}'.format(int(
    mean_squared_error(np.exp(y_test), np.exp(pred)))))
print('test rmse: {}'.format(int(
    mean_squared_error(np.exp(y_test), np.exp(pred), squared=False))))
print('test r2: {}'.format(
    r2_score(np.exp(y_test), np.exp(pred))))
print()

print('Average house price: ', int(np.exp(y_train).median()))