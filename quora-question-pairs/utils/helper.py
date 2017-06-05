'''
Created on May 21, 2017

@author: 703188429
'''
import numpy as np

def divideDataFrame(df, size):
    groups = df.groupby(np.arange(len(df.index))/size)
    frames = []
    for frameno, frame in groups:
        frames.append([frameno, frame])
    return frames

def removeUnwantedWordsAndConvertToLower(string):
    PERMITTED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ?" 
    return "".join(c for c in string if c in PERMITTED_CHARS).lower().replace('?', ' ? ').strip()
