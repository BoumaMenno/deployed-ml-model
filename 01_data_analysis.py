# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:00:10 2021

@author: Menno Bouma
"""
# imports
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import yaml

sns.set_theme(style="ticks")

# load dataset
data = pd.read_csv('data/train.csv')
print(data.shape)

# drop id column
data.drop('Id', axis=1, inplace=True)

## Target Variable ##
target = ['SalePrice']
data[target].hist(bins=50, density=True)
plt.ylabel('Number of houses')
plt.xlabel(target)
plt.show()

np.log(data[target]).hist(bins=50, density=True)
plt.ylabel('Number of houses')
plt.xlabel('Log of '+target[0])
plt.show()

""" Applying a log transform brings the target closer to gaussian """

## Explore data types and missing values ##
data.info()
# select categorical variables from feature set
categoricalFeatures = [feat for feat in data.columns if data[feat].dtype == 'O']
# add MSSubClass to categorical variables (from data_description.txt)
categoricalFeatures = categoricalFeatures + ['MSSubClass']
data['MSSubClass'] = data['MSSubClass'].astype('O')

# other variables are numeric
numericalFeatures = [feat for feat in data.columns if feat not in categoricalFeatures and feat != 'SalePrice']

# check which features have missing data
featuresWithMissing = [feat for feat in data.columns if data[feat].isnull().sum() > 0]
print(data[featuresWithMissing].isnull().mean().sort_values(ascending=False))

""" Several features with very high amount of missing values, these should be 
dropped. The missing values in the other features can be imputed later """

# save features with missing values 
featuresWithHighMissing = [feat for feat in featuresWithMissing if data[feat].isnull().mean() >= 0.3]
categoricalWithMissing = [feat for feat in categoricalFeatures if feat in featuresWithMissing ]
numericalWithMissing = [feat for feat in numericalFeatures if feat in featuresWithMissing ]

# make a division between marking data as not available or imputing the missing data with the most frequent category
imputeNA = [feat for feat in categoricalWithMissing if data[feat].isnull().mean() > 0.1]
imputeMostFrequent = [feat for feat in categoricalWithMissing if data[feat].isnull().mean() <= 0.1]

# impute missing values
data[imputeNA] = data[imputeNA].fillna('NA')
for feat in imputeMostFrequent:
    data[feat].fillna(data[feat].mode()[0], inplace=True)

for feat in numericalWithMissing:
    meanValue = data[feat].mean()
    data[feat + '_na'] = np.where(data[feat].isnull(), 1, 0)
    data[feat].fillna(meanValue, inplace=True)

print(data[featuresWithMissing].isnull().mean().sort_values(ascending=False))

## Explore Categorical Features ##
data[categoricalFeatures].nunique().sort_values(ascending=False).plot.bar(figsize=(12,5))

""" Low cardinality, so no action needed """

# from data_description.txt we can define the following mappings
basementExposureMap = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
garageFinishMap = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fenceMap = {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
qualityMap = {'NA': 0,'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
finishMap = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

data['BsmtExposure'] = data['BsmtExposure'].map(basementExposureMap)
data['GarageFinish'] = data['GarageFinish'].map(garageFinishMap)
data['Fence'] = data['Fence'].map(fenceMap)

qualityFeatures = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for feat in qualityFeatures:
    data[feat] = data[feat].map(qualityMap)

finishFeatures = ['BsmtFinType1', 'BsmtFinType2']
for feat in finishFeatures:
    data[feat] = data[feat].map(finishMap)
    
mappedCategoricalFeatures = qualityFeatures + finishFeatures + ['BsmtExposure','GarageFinish','Fence']

def categoryPlots(df, features):
    df = df[features+target].copy()
    df = df.melt(id_vars=target, var_name='Feature')
    sns.catplot(data=df, x='value', y=target[0], col='Feature', col_wrap=4, kind = 'box', sharex=False)
    plt.show()

categoryPlots(data, mappedCategoricalFeatures)

unmappedCategoricalFeatures = [ feat for feat in categoricalFeatures if feat not in mappedCategoricalFeatures ]

# Rare labels
def findRareLabels(df, feat, rare_perc):
    df = df.copy()

    # determine the % of observations per category
    perc = df.groupby(feat)[target[0]].count() / len(df)

    # return categories that are rare
    return perc[perc < rare_perc]

for feat in unmappedCategoricalFeatures:
    print(findRareLabels(data, feat, 0.01))
    print()
    
""" Several rare labels are found in the data, these should be removed as they tend to overfit. To do: what to do with rare labels? """

## Explore numerical features ##
# select year features
yearFeatures = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
referenceFeature = ['YrSold']

# define function to plot scatter of age of feature vs sale price
def scatterAge(df, yearFeat, refFeat):
    df = df.copy()
    
    df[yearFeat] = df[refFeat] - df[yearFeat]
    
    plt.scatter(df[yearFeat], df[target[0]])
    plt.ylabel(target[0])
    plt.xlabel(yearFeat)
    plt.show()
    
for feat in yearFeatures:
    scatterAge(data, feat, referenceFeature[0])
    
""" It looks like there is a negative correlation between ages and sale price, which makes sense"""

# select discrete features
print(data[numericalFeatures].nunique().sort_values(ascending=True))
discreteFeatures = [feat for feat in numericalFeatures if data[feat].nunique() < 15 and feat not in yearFeatures]
    
categoryPlots(data, discreteFeatures)

# select remaining numerical features
continuousFeatures = [feat for feat in numericalFeatures if feat not in discreteFeatures+yearFeatures]
data[continuousFeatures].hist(bins=30, figsize=(15,15))
plt.show()

""" Data is heavily skewed, and not all variables can be transformed using a log transformation. To do: which transformation to use? """
positiveContinuousFeatures = [feat for feat in continuousFeatures if (data[feat]<= 0).sum() == 0]
# log transformation on positive-only features
np.log(data[positiveContinuousFeatures]).hist(bins=30, figsize=(15,15))

# binarize heavily skewed features
heavilySkewedFeatures = ['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'MiscVal']

for feat in heavilySkewedFeatures:
    plt.figure()
    plt.title(feat)
    plt.xticks([0, 1], ['1', '0'])
    plt.hist(np.where(data[feat]==0, 0, 1), bins=2)

# yeo-johnson transformation on others
yeojohnsonFeatures = [feat for feat in continuousFeatures if feat not in positiveContinuousFeatures+heavilySkewedFeatures]

# save features to config
config = {  "CATEGORICAL_FEATURES" : categoricalFeatures,
            "CATEGORICAL_WITH_MISSING" : [ feat for feat in categoricalWithMissing if feat not in featuresWithHighMissing],
            "CATEGORICAL_WITH_HIGH_MISSING" : featuresWithHighMissing,
            "QUALITY_FEATURES" : qualityFeatures,
            "FINISH_FEATURES" : finishFeatures,
            "BASEMENT_EXPOSURE_FEATURES" : ['BsmtExposure'],
            "GARAGE_FINISH_FEATURES" : ['GarageFinish'],
            "FENCE_FEATURES" : ['Fence'],
            "QUALITY_MAP" : qualityMap,
            "FINISH_MAP" : finishMap,
            "BASEMENT_EXPOSURE_MAP" : basementExposureMap,
            "GARAGE_FINISH_MAP" : garageFinishMap,
            "FENCE_MAP" : fenceMap,
            "UNMAPPED_CATEGORICAL_FEATURES" : unmappedCategoricalFeatures,
            "NUMERICAL_FEATURES" : numericalFeatures,
            "NUMERICAL_WITH_MISSING" : numericalWithMissing,
            "YEAR_FEATURES" : yearFeatures,
            "REFERENCE_FEATURE" : referenceFeature,
            "POSITIVE_CONTINUOUS_FEATURES" : positiveContinuousFeatures,
            "HEAVILY_SKEWED_FEATURES" : heavilySkewedFeatures,
            "YEOJOHNSON_FEATURES" : yeojohnsonFeatures,
            "FEATURES" : categoricalFeatures+numericalFeatures,
              }

# dump to yaml
with open('config.yml', 'w') as outfile:
    yaml.dump(config, outfile)
