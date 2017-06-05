'''
Created on May 21, 2017

@author: 703188429
'''

from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
from ensembler import blending
from ensembler.optimization.tuningparams import grid_tuning
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import cross_validation 
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def getTrainHoldoutSplit(df, dfLabel, holdoutSize=0.15):
    dfHoldOut = None
    dfHoldOutLabel = None
    cv_pre = cross_validation.StratifiedShuffleSplit(dfLabel, 1, test_size=holdoutSize, random_state=0)
    for train_index, test_index in cv_pre:
        print("TRAIN:", train_index, "TEST:", test_index)
        print train_index.max(), test_index.max()
        print df.shape, type(df)
        y_train, y_test = dfLabel[train_index], dfLabel[test_index]
        x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        df, dfLabel = x_train, y_train
        dfHoldOut, dfHoldOutLabel = x_test, y_test
    print "==================== Data Set =================================="
    print "Holdout Set => ", dfHoldOut.shape
    print "Train Set => ", df.shape
    print "==================== Data Set =================================="
    return df, dfLabel, dfHoldOut, dfHoldOutLabel

def train():
    trainFile = os.path.join('data', 'quora_features_train.csv')
    df = pd.read_csv(trainFile)
    dfLabel = df['label']
    df.drop(['label'], axis=1, inplace=True)
    df, dfLabel, dfHoldOut, dfHoldOutLabel = getTrainHoldoutSplit(df, dfLabel)

    cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=3, test_size=0.2, random_state=1)
    clf = LogisticRegression()
    tuned_params = grid_tuning.LogisticRegressionTuningParams

    clf = RandomForestClassifier()
    tuned_params = grid_tuning.RandomForestClassifierTuningParams

    valcounts = dfLabel.value_counts()
    posBalanceWeight = valcounts[0]/float(valcounts[1])
    print posBalanceWeight
    clf = XGBClassifier()
    tuned_params = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        'n_estimators': [40, 100, 200, 500, 1000, 2000],
        'scale_pos_weight': [posBalanceWeight]
    }#grid_tuning.RandomForestClassifierTuningParams
    
    gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring="log_loss", n_jobs=6, error_score=-100)
    print df.shape, dfLabel.shape
    gscv.fit(df, dfLabel)
    print gscv.best_estimator_, gscv.best_score_
    print "HoldOut score LLs => ", metrics.log_loss(dfHoldOutLabel, gscv.best_estimator_.predict_proba(dfHoldOut))
    print "HoldOut score Acc => ", metrics.accuracy_score(dfHoldOutLabel, gscv.best_estimator_.predict(dfHoldOut))
    
    joblib.dump(gscv.best_estimator_, os.path.join("clf", "xgb.pickle"))
    #joblib.dump(gscv.best_estimator_, os.path.join("clf", gscv.best_estimator_.__name__))
    #submitcsv(gscv.best_estimator_)

if __name__ == "__main__":
    train()

