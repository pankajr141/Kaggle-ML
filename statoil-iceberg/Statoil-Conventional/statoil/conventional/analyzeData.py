'''
Created on Dec 29, 2017

@author: 703188429
'''
from __future__ import print_function
import os
import pandas as pd
import numpy as np

def exploreData(trainFile, testFile):
    print ("-------------- Train File -----------")
    dfTrain = pd.read_json(trainFile)
    for band in ['band_1', 'band_2']:
        data = dfTrain[band].as_matrix()
        data = np.array(map(lambda x: np.array(x), data)).flatten()
        print(band, '=> Min:', np.min(data), ', Max:', np.max(data))
        print(data.shape)
        bins, ranges = np.histogram(data, 10, (np.min(data), np.max(data)))   
        print('Bins: ', bins)
        print('Ranges:', ranges)

    print ("-------------- Test File -----------")
    dfTest = pd.read_json(testFile)
    for band in ['band_1', 'band_2']:
        data = dfTest[band].as_matrix()
        data = np.array(map(lambda x: np.array(x), data)).flatten()
        print(band, '=> Min:', np.min(data), ', Max:', np.max(data))
        print(data.shape)
        bins, ranges = np.histogram(data, 10, (np.min(data), np.max(data)))   
        print('Bins: ', bins)
        print('Ranges:', ranges)
    #print np.linspace(np.min(data), np.max(data), num=100)

if __name__ == "__main__":
    trainFile = os.path.join("..", "data", "train.json")
    testFile = os.path.join("..", "data", "test.json")
    exploreData(trainFile, testFile)
