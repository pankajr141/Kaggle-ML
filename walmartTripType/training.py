'''
Created on 19-Dec-2015

@author: pankajrawat
'''


from multiprocessing import freeze_support
import os

from scipy.optimize import minimize
from sklearn import ensemble, cross_validation
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import xgboost as xgb


pickle_dir = "E:\Research\Big Data\kaggle\walmart\preprocessing_objects"
dfMHoldOut_file = os.path.join(pickle_dir, "dfMHoldOut.pickle")
dfMHoldOutLabel_file = os.path.join(pickle_dir, "dfMHoldOutLabel.pickle")
dfM_file = os.path.join(pickle_dir, "dfM.pickle")
dfMTest_file = os.path.join(pickle_dir, "dfMTest.pickle")
dfMTestVisitNo_file = os.path.join(pickle_dir, "dfMTestVisitNo.pickle")
dfMLabel_file = os.path.join(pickle_dir, "dfMLabel.pickle")
SelectorModel_file = os.path.join(pickle_dir, "selectorModel.pickle")

headers = ['TripType_3', 'TripType_4', 'TripType_5', 'TripType_6',
               'TripType_7', 'TripType_8', 'TripType_9', 'TripType_12', 'TripType_14',
               'TripType_15', 'TripType_18', 'TripType_19', 'TripType_20', 'TripType_21',
               'TripType_22', 'TripType_23', 'TripType_24', 'TripType_25', 'TripType_26',
               'TripType_27', 'TripType_28', 'TripType_29', 'TripType_30', 'TripType_31',
               'TripType_32', 'TripType_33', 'TripType_34', 'TripType_35', 'TripType_36',
               'TripType_37', 'TripType_38', 'TripType_39', 'TripType_40', 'TripType_41',
               'TripType_42', 'TripType_43', 'TripType_44', 'TripType_999',
]


def preprocess(filename, train=True):
    dfM = pd.read_csv(filename)

    if train:
        dfM = dfM[dfM.FinelineNumber.notnull()]

    dfM['DepartmentDescription'] = dfM['DepartmentDescription'].fillna(method='ffill')
    if not train:
        dfM['TripType'] = dfM['VisitNumber']

    print len(dfM['VisitNumber'].unique())
    print "Creating dummies"
    x = dfM[['Weekday', 'DepartmentDescription']]
    x = pd.get_dummies(x)
    dfM = pd.concat([dfM, x], axis=1)

    print "Creating Weekend column"
    dfM['Weekend'] = dfM['Weekday'].apply(lambda x: 1 if x in ['Friday', 'Saturday', 'Sunday'] else 0)
    dfM.drop(['Weekday', 'DepartmentDescription'], axis=1, inplace=True)

    print "Groupby dummies"
    dfM_dummies = dfM.groupby(['VisitNumber'], as_index=False).max()
    if not train:
        joblib.dump(dfM_dummies['VisitNumber'], dfMTestVisitNo_file)

    dfM_dummies.drop(['ScanCount', 'TripType', 'VisitNumber', 'Upc', 'FinelineNumber'], axis=1, inplace=True)

    print "Groupby agg"
    itemPerVisit = lambda g: abs(g).sum()
    isItemReturned = lambda x: 1 if x.min() < 0 else 0
    itemsReturned = lambda x: x[x < 0].sum()  # it is panda series of that group by
    #percentItemBasedDept = lambda x: x[x > 0].sum() / dfM['']
    uniqueItem = lambda x: len(x.unique())

    f = {
            'TripType': 'max',
            'ScanCount': {
                           'rowscount': lambda x: x.count(),
                           'itemPerVisit': itemPerVisit,
                           'isitemReturned': isItemReturned,
                           'itemsReturned': itemsReturned,
                           #'%ItemBasedDept': percentItemBasedDept,
            },
            'Upc': {
                         'uniqueItem': uniqueItem,
            }
    }

    dfM = dfM.groupby(['VisitNumber'], as_index=False).agg(f)
    dfTmp1 = dfM['ScanCount']
    dfTmp2 = dfM['Upc']

    dfM.columns = dfM.columns.droplevel(level=1)

    dfM = pd.concat([dfM, dfTmp1, dfTmp2, dfM_dummies], axis=1)
    dfMLabel = dfM['TripType']
    dfM.drop(['ScanCount', 'TripType', 'VisitNumber'], axis=1, inplace=True)

    if not train:
        joblib.dump(dfM, dfMTest_file)
        print "Created test DF"
        return

    """Creating a hold out set for later validation"""
    print "Genrating HoldOutset"
    dfMHoldOut = None
    dfMHoldOutLabel = None
    print np.sort(dfMLabel.unique())
    cv_pre = cross_validation.StratifiedShuffleSplit(dfMLabel, 1, test_size=0.15, random_state=0)
    for train_index, test_index in cv_pre:
        print("TRAIN:", train_index, "TEST:", test_index)
        print train_index.max(), test_index.max()
        print dfM.shape, type(dfM)
        y_train, y_test = dfMLabel[train_index], dfMLabel[test_index]
        x_train, x_test = dfM.iloc[train_index], dfM.iloc[test_index]
        dfM, dfMLabel = x_train, y_train
        dfMHoldOut, dfMHoldOutLabel = x_test, y_test
        print np.sort(dfMHoldOutLabel.unique())
        print np.sort(dfMLabel.unique())
    #if np.sort(dfMLabel.unique()) != np.sort(dfMHoldOutLabel.unique()):
    #    print "ERR => Holdout set not created well"
    #    exit()
    joblib.dump(dfM, dfM_file)
    joblib.dump(dfMLabel, dfMLabel_file)
    joblib.dump(dfMHoldOut, dfMHoldOut_file)
    joblib.dump(dfMHoldOutLabel, dfMHoldOutLabel_file)


def process():
    dfM = joblib.load(dfM_file)
    dfMLabel = joblib.load(dfMLabel_file)
    dfMHoldOut = joblib.load(dfMHoldOut_file)
    dfMHoldOutLabel = joblib.load(dfMHoldOutLabel_file)

    print len(dfMHoldOutLabel.unique()), len(dfMLabel.unique())

    dfM.drop(['rowscount'], axis=1, inplace=True)
    dfMHoldOut.drop(['rowscount'], axis=1, inplace=True)

    dfMLabel = dfMLabel.as_matrix()
    dfM = dfM.as_matrix()
    dfMHoldOut = dfMHoldOut.as_matrix()

    print "Feature selection"
    if not os.path.exists(SelectorModel_file):
        fsmodel = ensemble.ExtraTreesClassifier().fit(dfM, dfMLabel)
        model = SelectFromModel(fsmodel, prefit=True)
        joblib.dump(model, SelectorModel_file)
    model = joblib.load(SelectorModel_file)

    dfM = model.transform(dfM)
    dfMHoldOut = model.transform(dfMHoldOut)

    print dfM.shape
    print dfMLabel.shape

    print dfM.shape, dfMHoldOut.shape
    #pca = PCA(n_components=1500)
    #dfM = pca.fit_transform(dfM)

    tuned_params = {
                        #'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        #'penalty': ['l2', 'l1'],
                        #'loss': ['deviance'],
                        #'learning_rate': [0.02, 0.01, 0.1, 0.05],
                        #'max_depth': [4, 6],
                        #'n_estimators': [500, 1000]
                   }

    clf = ensemble.AdaBoostClassifier()
    clf = ensemble.RandomForestClassifier()
    clf = ensemble.GradientBoostingClassifier()

    #clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    #clf = xgb.XGBClassifier()
    #clf = LogisticRegression()
    #log_scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False)

    # Pickled models
    clf = LogisticRegression(C=1.0, penalty='l2')
    clf = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5)

    print "Estimator => ", clf.__class__.__name__
    cv = cross_validation.StratifiedShuffleSplit(dfMLabel, n_iter=2, test_size=0.2, random_state=0)
    gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring="log_loss", n_jobs=1)
    gscv.fit(dfM, dfMLabel)

    estimator_pickefile = os.path.join(pickle_dir, gscv.best_estimator_.__class__.__name__)
    joblib.dump(gscv.best_estimator_, estimator_pickefile)

    print "Done"
    print gscv.best_estimator_, gscv.best_score_
    print "HoldOut score => ", metrics.log_loss(dfMHoldOutLabel, gscv.best_estimator_.predict_proba(dfMHoldOut))


def generateCSV():

    dfMHoldOut = joblib.load(dfMHoldOut_file)
    dfMHoldOutLabel = joblib.load(dfMHoldOutLabel_file)
    dfMHoldOut.drop(['rowscount'], axis=1, inplace=True)
    dfMHoldOut = dfMHoldOut.as_matrix()

    model = joblib.load(SelectorModel_file)
    dfMHoldOut = model.transform(dfMHoldOut)
    print dfMHoldOut.shape

    clf1 = joblib.load(os.path.join(pickle_dir, LogisticRegression().__class__.__name__))
    clf2 = joblib.load(os.path.join(pickle_dir, xgb.XGBClassifier().__class__.__name__))
    clfs = [clf1, clf2]
    predictions = []
    for clf in clfs:
        _prediction = clf.predict_proba(dfMHoldOut)
        print "Score: ", clf.__class__.__name__, " -> ", metrics.log_loss(dfMHoldOutLabel, _prediction)
        predictions.append(_prediction)

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight * prediction
        return metrics.log_loss(dfMHoldOutLabel, final_prediction)

    starting_values = [0.5] * len(predictions)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(predictions)
    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    print res
    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    exit()

    clf = joblib.load(os.path.join(pickle_dir, LogisticRegression().__class__.__name__))
    print metrics.log_loss(dfMHoldOutLabel, clf.predict_proba(dfMHoldOut))

    # CSV Generation code
    predictions = clf.predict_proba(dfMHoldOut)
    predictions = pd.DataFrame(predictions, columns=headers)
    predictions.to_csv(os.path.join(pickle_dir, 'result1.csv'))


def generateFinalCSV():

    dfM_visitnumber = joblib.load(dfMTestVisitNo_file)
    joblib.dump(dfM_visitnumber, dfMTestVisitNo_file)

    df = joblib.load(dfMTest_file)
    df1 = joblib.load(dfM_file)

    missing_column = df['rowscount'].apply(lambda x: 0)
    df.insert(39, 'DepartmentDescription_HEALTH AND BEAUTY AIDS', missing_column)
    print df.columns
    print df1.columns

    df.drop(['rowscount'], axis=1, inplace=True)
    df = df.as_matrix()
    print df.shape

    model = joblib.load(SelectorModel_file)
    df = model.transform(df)

    clf1 = joblib.load(os.path.join(pickle_dir, LogisticRegression().__class__.__name__))
    clf2 = joblib.load(os.path.join(pickle_dir, xgb.XGBClassifier().__class__.__name__))

    predictions = clf1.predict_proba(df)
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, 'VisitNumber', dfM_visitnumber)
    predictions.set_index(['VisitNumber'], inplace=True)
    predictions.columns = headers
    predictions.to_csv(os.path.join(pickle_dir, 'result1.csv'))


if __name__ == "__main__":
    trainFile = os.path.join('inputs', 'train.csv')
    testFile = os.path.join('inputs', 'test.csv')

    freeze_support()
    #preprocess(trainFile)
    #preprocess(testFile, train=False)
    #process()
    #generateCSV()
    generateFinalCSV()
