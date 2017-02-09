'''
Created on 15-Dec-2015

@author: pankajrawat
'''
import os

from sklearn.externals import joblib
import numpy as np
import pandas as pd

def cleanString(string):
    return ''.join(ch for ch in str(string) if ch.isalnum())


def _DiscreteVectorizer(df, feature, discoverNewFeatures=True, newFeatures=[], debug=False):
    if discoverNewFeatures:
        newFeatures = [feature + '_' + cleanString(uniqueId) for uniqueId in df[feature].unique()]

    #newDF = pd.DataFrame(np.zeros(shape=(len(df), len(newLabels))), columns=newLabels)
    newDF = pd.DataFrame(np.zeros(shape=(len(df), 1)), columns=["test"])
    debug_cntr = 0
    newDF = pd.DataFrame(np.zeros(shape=(len(df), len(newFeatures))), columns=newFeatures)
    matchString = []
    for nFeature in newFeatures: 
        matchString.append(cleanString(nFeature.split('_', 1)[1]))

    def __apply__():
        for nFeature in newFeatures: 
            newDF[nFeature] = df[feature].apply(lambda x: 1 if cleanString(x) == matchString else 0)
    df.apply(__apply__)

    for nFeature in newFeatures:
        matchString = cleanString(nFeature.split('_', 1)[1])
        newDF[nFeature] = df[feature].apply(lambda x: 1 if cleanString(x) == matchString else 0)
        debug_cntr += 1
        if debug_cntr % 50 == 0:
            print "Features processed => ", debug_cntr

    newDF = newDF.drop(["test"], axis=1)
    return newDF


def DiscreteVectorizer(df, features, dataFrame=True, discoverNewFeatures=True, newFeatures=[], debug=False):
    if not dataFrame:
        df = pd.DataFrame(df)
    for cntr, feature in enumerate(features):
        if debug:
            print "Expanding ", feature
        _newFeatures = [] if discoverNewFeatures else newFeatures[cntr]
        newDf = _DiscreteVectorizer(df, feature, discoverNewFeatures, _newFeatures, debug)
        df = pd.concat([df, newDf], axis=1)
        df = df.drop(feature, axis=1)
    return df
