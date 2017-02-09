'''
Created on 15-Dec-2015

@author: pankajrawat
'''
from datetime import datetime
import os
import time
from timeit import default_timer as timer

from sklearn import cross_validation 
from sklearn import ensemble
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

import convertion
from feature_extraction import DiscreteVectorizer, cleanString
import numpy as np
import pandas as pd


trainFile = os.path.join('inputs', 'train.csv')

dfMeta = pd.read_csv(trainFile)
dfMeta['UpcLen'] = dfMeta['Upc'].apply(lambda x: x if len(str(x)) < 5 else  float(str(x)[0:4]))
weekdayCols = ['Weekday' + '_' + cleanString(uniqueId) for uniqueId in dfMeta['Weekday'].unique()]
departmentDescriptionCols = ['DepartmentDescription' + '_' + cleanString(uniqueId) for uniqueId in dfMeta['DepartmentDescription'].unique()]
departmentDescriptionCols = filter(lambda x: x != 'DepartmentDescription_nan', departmentDescriptionCols)
finelineNumberCols = ['FinelineNumber' + '_' + cleanString(uniqueId) for uniqueId in dfMeta['FinelineNumber'].unique()]
finelineNumberCols = filter(lambda x: x != 'FinelineNumber_nan', finelineNumberCols)

chunksize = 50000


def createLargePickleObject(pickle_dir):
    pickleno = 0
    columns = []
    for i in range(0, len(dfMeta), chunksize):
        pickleno += 1
        dfFileTmp = os.path.join(pickle_dir, "dfTmp_" + str(pickleno) + ".pickle")
        print "Iteration => ", pickleno, " => ", dfFileTmp
        start = timer()
        if pickleno == 1:
            df = pd.read_csv(trainFile, nrows=chunksize, skiprows=i)
            columns = df.columns
        else:
            df = pd.read_csv(trainFile, nrows=chunksize, skiprows=i, header=None)
            df.columns = columns
        df['DepartmentDescription'] = df['DepartmentDescription'].fillna(method='ffill')

        df['FinelineNumber'] = df['FinelineNumber'].fillna(method='ffill')
        df['Upc'] = df['Upc'].fillna(method='ffill')

        if pickleno != 1:
            continue
        df = DiscreteVectorizer(df, ['Weekday', 'DepartmentDescription', 'FinelineNumber'], dataFrame=True, discoverNewFeatures=False,
                                newFeatures=[weekdayCols, departmentDescriptionCols, finelineNumberCols], debug=True)
        print df.shape
        joblib.dump(df, dfFileTmp)

        end = timer()
        delta = end - start
        print "Elapse time => ", delta


def groupbyLargeObjects(pickle_dir, groupby_dir):
    pickleno = 0
    for _ in range(0, len(dfMeta), chunksize):
        pickleno += 1
        dfFileTmp = os.path.join(pickle_dir, "dfTmp_" + str(pickleno) + ".pickle")
        print "Iteration => ", pickleno, " => ", dfFileTmp
        start = timer()

        df = joblib.load(dfFileTmp)
        # df = df.groupby(['VisitNumber', 'TripType'], as_index=False).agg(lambda x: 1 if x.sum() else 0)
        df = df.groupby(['VisitNumber', 'TripType'], as_index=False).max()
        dfFileTmp2 = os.path.join(groupby_dir, "dfGrouped_" + str(pickleno) + ".pickle")
        joblib.dump(df, dfFileTmp2)

        end = timer()
        delta = end - start
        print "Elapse time => ", delta


def getCombinedDf(groupby_dir):
    dfM = pd.DataFrame()
    dfMLabel = pd.DataFrame()
    pickleno = 0

    for _ in range(0, len(dfMeta), chunksize):
        pickleno += 1
        dfFileGroupby = os.path.join(groupby_dir, "dfGrouped_" + str(pickleno) + ".pickle")
        print "Iteration => ", pickleno, " => ", dfFileGroupby
        start = timer()
        df = joblib.load(dfFileGroupby)
        # dfLabel = df['TripType']
        # df = df.drop(['TripType'], axis=1)
        dfM = dfM.append(df)
        # dfMLabel = dfMLabel.append(dfLabel)
        del df
        end = timer()
        delta = end - start
        print "Elapse time => ", delta
    return dfM, dfMLabel



pickle_dir = "E:\Research\Big Data\kaggle\walmart\preprocessing_objects"
pickle_dir = "G:\kaggle\walmart\preprocessing_objects"
groupby_dir = "E:\Research\Big Data\kaggle\walmart\preprocessing_objects"

# createLargePickleObject(pickle_dir)
# groupbyLargeObjects(pickle_dir,groupby_dir)
# dfM, dfMLabel = getCombinedDf(groupby_dir)
# joblib.dump(dfM, os.path.join(groupby_dir, "DFfinal.pickle"))
dfM = joblib.load(os.path.join(groupby_dir, "DFfinal.pickle"))
dfMLabel = dfM['TripType']
dfM['ScanCount'] = dfM['ScanCount'].apply(lambda x: abs(x))
dfM = dfM.drop(['TripType', 'VisitNumber', 'Upc'], axis=1)


print dfM.shape
print dfM.columns

dfMLabel = dfMLabel.as_matrix()
dfM = dfM.as_matrix()


print "Feature selection"
time.sleep(5)
print "sleeping"
fsmodel = ensemble.ExtraTreesClassifier().fit(dfM, dfMLabel)
model = SelectFromModel(fsmodel, prefit=True)
dfM = model.transform(dfM)
print dfM.shape

# pca = PCA(n_components=1500)
# dfM = pca.fit_transform(dfM)


cv = cross_validation.StratifiedShuffleSplit(dfMLabel, 3, test_size=0.2, random_state=0)
tuned_params = {}
clf = ensemble.RandomForestClassifier()
clf = ensemble.AdaBoostClassifier()
# clf = ensemble.GradientBoostingClassifier()
# log_scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False)
gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring="log_loss")
gscv.fit(dfM, dfMLabel)
print "Done"
print gscv.best_estimator_, gscv.best_score_

exit()

#===============================================================================
# dfvv = pd.read_csv(trainFile)
# 
# print dfvv.memory_usage(index=True).sum()/ (1024 * 1024)
# dfvv = dfvv.groupby(['VisitNumber', 'TripType'],  as_index=False).sum()
# print dfvv
# print dfvv.memory_usage(index=True).sum()/ (1024 * 1024)
#===============================================================================

#===============================================================================
# def uniqueCount(x):
#     return np.size(np.unique(x))
# dfvv = dfvv.groupby(['Weekday', 'TripType'], as_index=False)['VisitNumber'].agg(uniqueCount)
# print dfvv
# print dfvv.shape
#===============================================================================


# print len(dfvv['DepartmentDescription'].unique())
dfFile = os.path.join('objects', 'df.pickle')
dfLabelFile = os.path.join('objects', 'dfLabel.pickle')

def preProcessTrainData():
    df = pd.read_csv(trainFile)
    dfLabel = df['TripType']
    df = df.drop(['TripType'], axis=1)
    #===========================================================================
    # df['Weekday'] = df['Weekday'].apply(lambda x: convertion.weekdayDict[x])
    # df['DepartmentDescriptionTmp'] = df['DepartmentDescription'].shift()
    #===========================================================================

    # Replace null occurances with previous valid values
    df['DepartmentDescription'].fillna(method='ffill', inplace=True)
    df['FinelineNumber'] = df['FinelineNumber'].fillna(method='ffill')
    df['Upc'] = df['Upc'].fillna(method='ffill')

    df = DiscreteVectorizer(df, ['Weekday', 'DepartmentDescription'], dataFrame=True)
    joblib.dump(df, dfFile)
    joblib.dump(dfLabel, dfLabelFile)

# preProcessTrainData()

df = joblib.load(dfFile)
dfFile = os.path.join('objects', 'df_extended.pickle')
df = DiscreteVectorizer(df, ['FinelineNumber'], dataFrame=True)
joblib.dump(df, dfFile)

exit()
dfLabel = joblib.load(dfLabelFile)
df['Upc'] = df['Upc'].fillna(method='ffill')
df['UpcLen'] = df['Upc'].apply(lambda x: x if len(str(x)) < 5 else  float(str(x)[0:4]))
print df

# print df[(df['Upc'] > 100000000) & (df['Upc'] < 999999999)].shape
# print df[df['Upc'] > 999999999].shape
print df.shape
df = df.drop(['Upc'], axis=1)
# df  = df.drop(['VisitNumber'], axis=1)
# df['ScanCount'] = df['ScanCount'].apply(lambda x: abs(x))


# x_train, x_test, y_train, y_test = cross_validation.train_test_split(df, dfLabel)
#===============================================================================
# print df.shape
# for train_index, test_index in cv:
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = df[train_index], df[test_index]
#     y_train, y_test = dfLabel[train_index], dfLabel[test_index]
#     print x_train.shape, x_test.shape
# exit()
#===============================================================================

exit()
def unique_classifier():
    clf = ensemble.RandomForestClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    y_predicted = clf.predict_proba(x_test)
    print clf, metrics.log_loss(y_test, y_predicted)

unique_classifier()

exit()  # ## EXIT

x_train, x_test, y_train, y_test = cross_validation.train_test_split(df, dfLabel)


def getClassifiersList():
    from sklearn import ensemble
    from sklearn import neighbors
    from sklearn import naive_bayes

    classifiers = [
                        naive_bayes.GaussianNB(),
                        naive_bayes.MultinomialNB(),
                        ensemble.AdaBoostClassifier(),
                        ensemble.GradientBoostingClassifier(),
                        ensemble.RandomForestClassifier(),
                        ensemble.BaggingClassifier(),
                        neighbors.KNeighborsClassifier()
    ]
    return classifiers

for clf in getClassifiersList():
    pca = PCA(n_components=10)
    x_train = pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    # print metrics.classification_report(y_test, predicted)
    # print metrics.confusion_matrix(y_test, predicted)
    y_predicted = clf.predict_proba(x_test)
    # print y_predicted
    # print y_predicted.shape
    print clf, metrics.log_loss(y_test, y_predicted)
