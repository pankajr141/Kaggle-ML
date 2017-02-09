'''
Created on 09-Nov-2015

@author: pankajrawat
'''

from multiprocessing import Process, freeze_support
import os

from sklearn import cross_validation
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV

import numpy as np
import pandas as pd


def train_model(classifier_pickle_path, pca_pickle_path):
    df = pd.load(os.path.join('objects', 'df.pickle'))
    df_sales = pd.load(os.path.join('objects', 'df_sales.pickle'))

    df['Date'] = df['Date'].apply(lambda x: x.timetuple().tm_mon)

    df = df.as_matrix()
    df_sales = df_sales.as_matrix()
    pca = PCA(n_components=10)
    pca.fit(df)
    pca_df = pca.transform(df)
    split_from = 50000
    x_test = pca_df[:split_from]
    x_train = pca_df[split_from:]
    y_test = df_sales[:split_from]
    y_train = df_sales[split_from:]

    #  Single train_test_split validation
    clf = ensemble.RandomForestRegressor(n_jobs=3)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print "Train score: ", clf.score(x_train, y_train), " Test score: ", clf.score(x_test, y_test)
    joblib.dump(clf, classifier_pickle_path)
    joblib.dump(pca, pca_pickle_path)
    return

    # Comment above
    tuned_parameters = {
                            'n_estimators': np.linspace(20, 500, 2, endpoint=True).astype(int),
                            #'min_samples_split': np.linspace(1, 1000, 2, endpoint=True).astype(int),
                            'min_samples_leaf': [10, 20],
                        }
    print tuned_parameters
    clf = ensemble.RandomForestRegressor(n_jobs=3)
    gscv = GridSearchCV(clf, param_grid=tuned_parameters, cv=cross_validation.train_test_split(x_train, y_train, test_size=0.1), verbose=3)
    gscv.fit(x_train, y_train)
    for params, mean_score, scores in gscv.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    best_estimator = gscv.best_estimator_
    print best_estimator
    print(gscv.best_params_)


    #clf.fit(df1, df_sales)
    #print "Feature importance: ", clf.feature_importances_
    predicted = gscv.predict(x_test)
    #print predicted[predicted > 0][0:50]
    print "Less than 0 => ", predicted[predicted < 0]
    #np.savetxt('Output.csv', predicted, delimiter =' ')
    print "Train score: ", gscv.score(x_train, y_train)
    print "Test score: ", gscv.score(x_test, y_test)
    #joblib.dump(gscv.best_estimator_, classifier_pickle_path)


def test_model(classifier_pickle_path, pca_pickle_path):
    clf = joblib.load(classifier_pickle_path)
    pca = joblib.load(pca_pickle_path)
    df_test = pd.load(os.path.join('objects', 'df_test.pickle'))
    df_test = df_test.drop(['Id'], axis=1)
    df_test['Date'] = df_test['Date'].apply(lambda x: x.timetuple().tm_mon)
    df_test = df_test.as_matrix()
    pca_df = pca.transform(df_test)
    predicted = clf.predict(pca_df)
    predicted = predicted.astype(int)
    id_array = np.arange(1, 41089)
    predicted = np.column_stack([id_array, predicted])
    print predicted
    np.savetxt(os.path.join('objects', 'output.csv'), predicted, delimiter=',')


if __name__ == '__main__':
    freeze_support()
    classifier_pickle_path = os.path.join("classifier", "randomForestClassifier.pkl")
    pca_pickle_path = os.path.join("pca", "pca.pkl")
    #train_model(classifier_pickle_path, pca_pickle_path)
    test_model(classifier_pickle_path, pca_pickle_path)
