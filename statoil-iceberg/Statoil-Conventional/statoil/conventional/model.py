'''
Created on Dec 29, 2017

@author: 703188429
'''
from __future__ import print_function

import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation 
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
#import lightgbm
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np

from statoil.conventional import featureEngineering as fe

def getTrainHoldoutSplit(df, dfLabel, holdoutSize=0.15):
    dfHoldOut = None
    dfHoldOutLabel = None
    cv_pre = cross_validation.StratifiedShuffleSplit(dfLabel, 1, test_size=holdoutSize, random_state=10)
    for train_index, test_index in cv_pre:
        print("TRAIN:", train_index, "TEST:", test_index)
        print(train_index.max(), test_index.max())
        print(df.shape, type(df))
        y_train, y_test = dfLabel[train_index], dfLabel[test_index]
        x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        df, dfLabel = x_train, y_train
        dfHoldOut, dfHoldOutLabel = x_test, y_test
    print("==================== Data Set ==================================")
    print("Holdout Set => ", dfHoldOut.shape)
    print("Train Set => ", df.shape)
    print("==================== Data Set ==================================")
    return(df, dfLabel, dfHoldOut, dfHoldOutLabel)


def createModel(trainFile, testFile):
    dfH = fe.getHistogramsBins(trainFile, testFile, dataset='train', bins=40, density=False)
    dfS = fe.getStructuralFeatures(trainFile)
    dfW = fe.generateWindowsBins(trainFile, flip=None)
    df = pd.concat([dfH, dfS, dfW], axis=1)

    #df = df.head(100)
    print(df.shape)
    print(df.columns)
    dfLabel = df['is_iceberg']
    df, dfLabel, dfHoldOut, dfHoldOutLabel = getTrainHoldoutSplit(df, dfLabel, holdoutSize=0.20)


    """ Generating Flip Images """
    dfW_Flip_H = fe.generateWindowsBins(trainFile, flip=1)
    dfW_Flip_V = fe.generateWindowsBins(trainFile, flip=0)

    """ Since Flip images is duplicate images, making sure to filter any HoldOut set image from this data set. Sorting done since shuffle split rearranges index """
    dfFlipLabel = dfLabel.sort_index(ascending=True)
    dfW_Flip_H = dfW_Flip_H[~dfW_Flip_H.index.isin(dfHoldOutLabel.index.tolist())]
    dfW_Flip_V = dfW_Flip_V[~dfW_Flip_V.index.isin(dfHoldOutLabel.index.tolist())]
    dfH_Flip = dfH[~dfH.index.isin(dfHoldOutLabel.index.tolist())]
    dfS_Flip = dfS[~dfS.index.isin(dfHoldOutLabel.index.tolist())]

    df_Flip_H = pd.concat([dfH_Flip, dfS_Flip, dfW_Flip_H], axis=1)
    df_Flip_V = pd.concat([dfH_Flip, dfS_Flip, dfW_Flip_V], axis=1)
    
    '''
    df = pd.concat([df, df_Flip_H, df_Flip_V], axis=0)
    dfLabel = pd.concat([dfLabel, dfFlipLabel, dfFlipLabel])
    '''
    #df = pd.concat([df, df_Flip_H], axis=0)
    #dfLabel = pd.concat([dfLabel, dfFlipLabel])
    print("Shape after joining vertical and horizontal flips", df.shape, dfLabel.shape)
 
    df.drop(['id', 'is_iceberg', 'band_1', 'band_2'], axis=1, inplace=True)
    dfHoldOut.drop(['id', 'is_iceberg', 'band_1', 'band_2'], axis=1, inplace=True)

    '''
    pca = PCA(n_components=df.shape[1], svd_solver='randomized', whiten=True).fit(df)
    df = pca.transform(df)
    dfHoldOut = pca.transform(dfHoldOut)
    '''
    
    std_scale = preprocessing.StandardScaler().fit(df)
    df = std_scale.transform(df)
    dfHoldOut = std_scale.transform(dfHoldOut)

    cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=4, test_size=0.20, random_state=6)

    clf = LogisticRegression()
    tuned_params = {
                    'C': [0.001, 0.01, 0.05, 0.02, 0.1, 1.0],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                    'penalty': ['l1', 'l2'],    
    }


    clf = svm.SVC(kernel='poly', probability=True)
    tuned_params = {'C': np.logspace(-4, 1, 100),
              'gamma': np.logspace(-4, 1, 100), 
              'degree': [2,3,4]
    }

    '''
    clf = lightgbm.sklearn.LGBMClassifier()
    tuned_params = {
        'nthread': [3],
        #'learning_rate': np.logspace(-4, 1, 20),
        'n_estimators': [40, 60, 80, 100, 200, 250, 400],
        #'num_leaves': [5, 10, 15, 20, 25, 31, 40]
    }
    '''
    clf = xgb.XGBClassifier()    
    tuned_params = {
        'nthread': [3],
        'n_estimators': [40, 60, 80, 100, 200, 250, 300],
    }

    #tuned_params = {}
    #     clf = KNeighborsClassifier()
    #     tuned_params = {
    #         'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12],
    #         'leaf_size': [10,20,30,40]
    #     }

    print("Estimator => ", clf.__class__.__name__)
    pd.DataFrame(df).to_csv("data.csv")
    gscv = grid_search.GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=4, scoring="log_loss", n_jobs=5, error_score=-100)
    #gscv = grid_search.RandomizedSearchCV(clf, param_distributions=tuned_params, n_iter=100, cv=cv, verbose=4, scoring="log_loss", n_jobs=10, error_score=-100)
    
    print(df.shape, dfLabel.shape)
    gscv.fit(df, dfLabel)
    print(gscv.best_estimator_, gscv.best_score_)
    print("Train score LLs => ", metrics.log_loss(dfLabel, gscv.best_estimator_.predict_proba(df)))
    print("Train score Acc => ", metrics.accuracy_score(dfLabel, gscv.best_estimator_.predict(df)))
    print("HoldOut score LLs => ", metrics.log_loss(dfHoldOutLabel, gscv.best_estimator_.predict_proba(dfHoldOut)))
    print("HoldOut score Acc => ", metrics.accuracy_score(dfHoldOutLabel, gscv.best_estimator_.predict(dfHoldOut)))
    joblib.dump(gscv.best_estimator_, os.path.join("..", "clf", "xgb.pickle"))
    #joblib.dump(gscv.best_estimator_, os.path.join("clf", gscv.best_estimator_.__name__))
    return gscv.best_estimator_, std_scale

def submitTestCSV(trainFile, testFile, model, std_scale):
    df = fe.getHistogramsBins(trainFile, testFile, dataset='test', bins=40, density=False)
    dfS = fe.getStructuralFeatures(testFile)
    dfW = fe.generateWindowsBins(testFile, flip=None)
    df = pd.concat([df, dfS, dfW], axis=1)
    #df = df.head(100)
    print(df.shape)
    print(df.columns)
    dfId = df['id']
    df.drop(['id', 'band_1', 'band_2'], axis=1, inplace=True)
    df = std_scale.transform(df)
    print(df)
    output = model.predict_proba(df)
    dfO = pd.DataFrame(output, columns=['is_not_iceberg', 'is_iceberg'])
    df = pd.concat([dfId, dfO], axis=1)
    df = df[['id', 'is_iceberg']]
    df.to_csv("sub.csv", index=False)
    print(df.shape)

if __name__ == "__main__":
    trainFile = os.path.join("..", "data", "train.json")
    testFile = os.path.join("..", "data", "test.json")
    print(os.path.exists(trainFile))
    model, std_scale = createModel(trainFile, testFile)
    #model = joblib.load(os.path.join("..", "clf", "xgb.pickle"))
    submitTestCSV(trainFile, testFile, model, std_scale)
 