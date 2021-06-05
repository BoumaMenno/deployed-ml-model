# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:00:10 2021

@author: Menno Bouma
"""
# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

import scipy.stats as stats

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

""" Applying a log transform brings the target closer to guassian """


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
featuresWithHighMissing = [feat for feat in featuresWithMissing if data[feat].isnull().mean() <= 0.75]
categoricalWithMissing = [feat for feat in categoricalFeatures if (feat in featuresWithMissing) & (feat not in featuresWithHighMissing)]
numericalWithMissing = [feat for feat in numericalFeatures if (feat in featuresWithMissing) & (feat not in featuresWithHighMissing)]

## Explore Categorical Features ##
data[categoricalFeatures].nunique().sort_values(ascending=False).plot.bar(figsize=(12,5))

# from data_description.txt we can define the following mappings
basementExposureMap = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, 'Missing': 0, 'NA': 0}
garageFinishMap = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fenceMap = {'nan':0, 'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
qualityMap = {'Missing': 0, 'NA': 0,'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
finishMap = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

data['BsmtExposure'] = data['BsmtExposure'].map(basementExposureMap)
data['GarageFinish'] = data['GarageFinish'].map(garageFinishMap)
data['Fence'] = data['Fence'].map(fenceMap)

qualityFeatures = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for feat in qualityFeatures:
    data[feat] = data[feat].map(qualityMap)

finishFeatures = ['BsmtFinType1', 'BsmtFinType2']
for feat in finishFeatures:
    data[feat] = data[feat].map(finishMap)
    
allQualityFeatures = qualityFeatures + finishFeatures + ['BsmtExposure','GarageFinish','Fence']


qualityData = data[allQualityFeatures+target]
sns.catplot(x=allQualityFeatures, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)

def categoryPlots(df, features):
    df = df[features+target].copy()
    df = df.melt(id_vars=target, var_name='Feature')
    sns.catplot(data=df, x='value', y=target[0], col='Feature', col_wrap=4, kind = 'box')
    plt.show()

categoryPlots(data, allQualityFeatures)

# Rare labels

## Explore numerical features ##
# select year features
yearFeatures = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']

# define function to plot scatter of age of feature vs sale price
def scatterAge(df, yearFeat):
    df = df.copy()
    
    df[yearFeat] = df['YrSold'] - df[yearFeat]
    
    plt.scatter(df[yearFeat], df[target])
    plt.ylabel(target)
    plt.xlabel(yearFeat)
    plt.show()
    
for feat in yearFeatures:
    if feat != 'YrSold':
        scatterAge(data, feat)
    
""" It looks like there is a negative correlation between ages and sale price, which makes sense"""

# select discrete features
print(data[numericalFeatures].nunique().sort_values(ascending=True))
discreteFeatures = [feat for feat in numericalFeatures if data[feat].nunique() < 15 and feat not in yearFeatures]
    
for feat in discreteFeatures:
    # make boxplot with Catplot
    sns.catplot(x=feat, y=target[0], data=data, kind="box", height=4, aspect=1.5)
    plt.show()

# select remaining numerical features
continuousFeatures = [feat for feat in numericalFeatures if feat not in discreteFeatures+yearFeatures]
data[continuousFeatures].hist(bins=30, figsize=(15,15))
plt.show()

""" Data is heavily skewed, and not all variables can be transformed using a log transformation. To do: which transformation to use? """

