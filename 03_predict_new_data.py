# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:31:39 2021

@author: Menno
"""
import pandas as pd
import numpy as np

import joblib
import yaml

with open('config.yml') as infile:
    config = yaml.load(infile, Loader=yaml.FullLoader)

with open('pipeline.joblib', 'rb') as infile:
    pipeline = joblib.load(infile)

data = pd.read_csv('data/test.csv')
data.drop('Id', axis=1, inplace=True)
data['MSSubClass'] = data['MSSubClass'].astype('O')

TRAIN_FEATURES_WITH_MISSING = config["CATEGORICAL_WITH_MISSING"] + config["CATEGORICAL_WITH_HIGH_MISSING"] + config["NUMERICAL_WITH_MISSING"]
TEST_FEATURES_WITH_MISSING = [feat for feat in config["FEATURES"] if feat not in TRAIN_FEATURES_WITH_MISSING and data[feat].isnull().sum() > 0]

data[TEST_FEATURES_WITH_MISSING].isnull().mean()
data.dropna(subset=TEST_FEATURES_WITH_MISSING, inplace=True)

pred_test = np.exp(pipeline.predict(data))
pd.Series(pred_test).hist(bins=30)