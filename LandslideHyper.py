# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:52:30 2021

@author: HamishMitchell
"""
# IMPORTS

import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Import baseline df for subsidence
df = pd.read_csv("LAND_ImputeEncodedf.csv", index_col=['x', 'y'])
# Separate target and the predictors 
y = df.loc[:, 'Landslide']
X = df.drop('Landslide', axis=1)
print(y.shape)
print(X.shape)

# Train test split
X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=0)

# Model
dtc = DecisionTreeClassifier(random_state = 0)
print("Parameters currently in use:", dtc.get_params())
dtc.fit(X_train, y_train)


# Evaluation procedure; check for overfitting
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
# Evaluate model
scores = cross_val_score(dtc, X_temp, y_temp, scoring='roc_auc', cv=cv, n_jobs=-1)
# Summarize performance
print("Scores:", scores)
print("Mean ROC AUC:", scores.mean(), scores.std())


# Confusion matrix
y_temp_pred = cross_val_predict(dtc, X, y, cv=10)
print("Baseline confusion matrix:", confusion_matrix(y, y_temp_pred))

# Hyperparameter tuning
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(dtc, params, scoring='roc_auc', verbose=1, cv=10)
grid_search_cv.fit(X, y)
print("Scores:", grid_search_cv.cv_results_)
print("Best parameters for Decision Tree Classifier:", grid_search_cv.best_params_)
print("Best score for the Decision Tree:", grid_search_cv.best_score_)

landProb = grid_search_cv.predict_proba(X)[:, 1] # call estimator with the best hyperparameters found to get probabilities
probabilisticMap = pd.DataFrame(landProb.flatten(), columns=['Landslide_Probability'], index=X.index)
probabilisticMap.to_csv("LandslideProbabilities.csv")
