# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:35:09 2021

@author: HamishMitchell
"""

# Preprocessing 
import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV


# Load subsidence dataset
df = pd.read_csv("NewSubsImputeEncode.csv", index_col=['x', 'y'])
print(df.shape)
baseDF = df[df['GroundMotionRATE_m/yr'].notnull()]
print(baseDF.shape)


# Separate target and the predictors 
y = baseDF.loc[:, 'GroundMotionRATE_m/yr']
X = baseDF.drop('GroundMotionRATE_m/yr', axis=1)
print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Model
rfr = RandomForestRegressor(random_state = 42)
print(rfr.get_params())

# Cross validation procedure
scores = cross_val_score(rfr, X_train, y_train, scoring='r2', cv=10)
print("Cross Validation:\n")
print(scores)
print(scores.mean(), scores.std())
print('-'*40)


# Test on the verification set
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

# Evaluation metrics
print("Verification set:\n")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred))) 
print('-'*40)

# HYPERPARAMETER TUNING
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rfr_random = RandomizedSearchCV(estimator=rfr, param_distributions=random_grid, scoring='r2', n_iter=10, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rfr_random.fit(X_train, y_train)
print("Randomised Search Cross Validation:\n")
print("Scores:", rfr_random.cv_results_)
print("Best parameters for Radom Forest Regressor:", rfr_random.best_params_)
print("Best score for the Radom Forest Regressor:", rfr_random.best_score_)
print('-'*40)

# Load subsidence dataset
y = df.loc[:, 'GroundMotionRATE_m/yr']
X = df.drop('GroundMotionRATE_m/yr', axis=1)

# Predict the rest of the data
RandRF_predY = rfr_random.predict(X)
predict_y = pd.DataFrame(RandRF_predY.flatten(), columns=['GroundMotionRATE_m/yr'], index=X.index)
predict_y.to_csv('SubsidenceALLswaths.csv')


