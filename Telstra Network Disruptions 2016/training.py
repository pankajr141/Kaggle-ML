'''
Created on 18-Jan-2016

@author: pankajrawat
'''
import datetime
from multiprocessing import freeze_support
import os

from sklearn import cross_validation
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ensembler import blend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


pickle_dir = 'objects'


def preprocess():
    print "Starting time => ", datetime.datetime.now()
    global df
    global dfLabel
    global dfTest
    global dfTestId
    global dfHoldOut
    global dfHoldOutLabel

    df = joblib.load(os.path.join(pickle_dir, 'df.pickle'))
    dfLabel = joblib.load(os.path.join(pickle_dir, 'dfLabel.pickle'))
    dfTest = joblib.load(os.path.join(pickle_dir, 'dfTest.pickle'))
    #print dfTest[dfTest['locI_1'] == 0]['id']
    dfId = df['id']
    dfTestId = dfTest['id']

    df.drop(['id'], axis=1, inplace=True)
    dfTest.drop(['id'], axis=1, inplace=True)
    columns_to_drop = []
    for column in df.columns:
        #if filter(lambda x: column.startswith(x), ['loc_', 'lf_', 'st_', 'rt_', 'et_']):
        if filter(lambda x: column.startswith(x), ['locI_']):
            columns_to_drop.append(column)

    df.drop(columns_to_drop, axis=1, inplace=True)
    dfTest.drop(columns_to_drop, axis=1, inplace=True)
    print df.columns

    print "'Train columns = Test Columns ' : ", np.array_equal(df.columns, dfTest.columns)

    dfHoldOut = None
    dfHoldOutLabel = None
    cv_pre = cross_validation.StratifiedShuffleSplit(dfLabel, 1, test_size=0.15, random_state=0)
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


def visualize():
    print df.shape
    pca = PCA(n_components=10)
    y = pca.fit_transform(df)
    print y[:, 1]
    plt.plot(y[:, 0], y[:, 1])
    plt.show()
    exit()


def features_selection():
    global df, dfTest, dfLabel, dfHoldOut
    fsmodel = ExtraTreesClassifier().fit(df, dfLabel)
    model = SelectFromModel(fsmodel, prefit=True)
    df = model.transform(df)
    dfTest = model.transform(dfTest)
    dfHoldOut = model.transform(dfHoldOut)
    print df.shape


def train():
    #===========================================================================
    # clf = GradientBoostingClassifier()
    # clf = KNeighborsClassifier()
    # clf = RandomForestClassifier()
    # clf = ExtraTreesClassifier(n_estimators=2000, max_features=None, max_depth=7)
    # clf = SVC(C=1.0, gamma=0.001, kernel='linear', probability=True)
    # clf = LogisticRegression()
    # clf = RandomForestClassifier()
    #===========================================================================
    clf = xgb.XGBClassifier()

    print "Estimator => ", clf.__class__.__name__

    tuned_params = {
                    #'learning_rate': [0.001, 0.01, 0.05, 0.02, 0.1, 1.0],
                    #'n_estimators': [400, 1000, 2000, 3000],
                    #'max_depth': [3, 4, 5],
                    'nthread': [3]
                    #'loss': ['deviance', 'exponential'],
                    #'max_features': ["sqrt"]
                    #'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                    #'algorithm': ['auto', 'brute'],
                    #'n_neighbors': [50, 60, 70, 100, 150],
                    #'p': [1, 2],
                    #'weights': ['distance', 'uniform'],
                    #'n_estimators': [100, 1000, 2000, 5000, 10000],
                    #'max_features': [None, 'sqrt', 'log2'],
                    # 'penalty': ['l1', 'l2'],
                    # 'C': [0.01, 1.0, 10.0, 100.0],
                    #'gamma': ['auto', .01, 0.001, 0.0001],
                    #'kernel': ['linear'],
                    #'degree': [2, 3]
    }

    cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=3, test_size=0.2, random_state=1)
    gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring="log_loss", n_jobs=1)
    print df.shape, dfLabel.shape
    gscv.fit(df, dfLabel)
    print gscv.best_estimator_, gscv.best_score_
    print "HoldOut score LLs => ", metrics.log_loss(dfHoldOutLabel, gscv.best_estimator_.predict_proba(dfHoldOut))
    print "HoldOut score Acc => ", metrics.accuracy_score(dfHoldOutLabel, gscv.best_estimator_.predict(dfHoldOut))
    submitcsv(gscv.best_estimator_)
    #estimator_pickefile = os.path.join(pickle_dir, gscv.best_estimator_.__class__.__name__)
    #joblib.dump(gscv.best_estimator_, estimator_pickefile)


# Function to generate CSV
def submitcsv(clf=None):
    print "Submitting CSV classifier argument : ", clf
    headers = ['predict_0', 'predict_1', 'predict_2']
    if not clf:
        clf = joblib.load(os.path.join(pickle_dir, xgb.XGBClassifier().__class__.__name__))
    predictions = clf.predict_proba(dfTest)
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, 'id', dfTestId)
    predictions.set_index(['id'], inplace=True)
    predictions.columns = headers
    if not os.path.exists("output"):
        os.makedirs("output")
    predictions.to_csv(os.path.join('output', 'result.csv'))


def blending():
    # Classifiers to use in blending
    clfs = [
                    RandomForestClassifier(n_estimators=400, n_jobs=3, max_depth=6, max_features=None),
                    SVC(probability=True, degree=3, gamma=0.001, kernel='linear'),
                    xgb.XGBClassifier(n_estimators=400, nthread=3, learning_rate=0.1, max_depth=5),
                    ExtraTreesClassifier(max_depth=6, n_estimators=1000, max_features=None),
                    KNeighborsClassifier(algorithm='brute', metric='manhattan', n_neighbors=100, weights='distance'),
                    GradientBoostingClassifier(max_features=None, learning_rate=0.05, loss='deviance', n_estimators=1000)
    ]

    metaTunedParamsXGB = {
                          'learning_rate': [0.001, 0.05, 0.02, 0.01, 0.1, 0.5, 1.0],
                          'n_estimators': [100, 400, 1000, 2000],
                          'max_depth': [3, 4, 6]
                        }
    metaTunedParamsXGB = {
                          'learning_rate': [0.01],
                          'n_estimators': [2000],
                          'max_depth': [4]
                        }
    metaEstimator = xgb.XGBClassifier(nthread=1)
    metaTunedParams = metaTunedParamsXGB

    blendModel = blend.BlendModel(clfs, nFoldsBase=3, saveAndPickBaseDump=True, saveAndPickBaseDumpLoc=r'E:\workspace\challenges\telstra\tmp1',
                                             metaEstimator=metaEstimator,
                                             metaTunedParams=metaTunedParams
                                             )
    blendModel.fit(df, dfLabel)
    blendModel.score(dfHoldOut, dfHoldOutLabel)
    submitcsv(blendModel)


if __name__ == "__main__":
    # Set to 1 to enable blending
    blending = 0
    freeze_support()
    preprocess()
    #visualize()
    #features_selection()  # uncomment to reduce features
    if not blending:
        train()
        exit()
    blending()
