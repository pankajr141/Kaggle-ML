'''
Created on Dec 29, 2017

@author: 703188429
'''
from __future__ import print_function
import os
import pandas as pd
import numpy as np
from skimage.util.shape import view_as_windows
import cv2

def getHistogramsBins(trainFile, testFile, dataset='train', density=False, bins=50):
    dfTrain = pd.read_json(trainFile)
    data1 = dfTrain['band_1'].as_matrix()
    data1 = np.array(map(lambda x: np.array(x), data1))
    dataFlatten1 = data1.flatten()

    band1_min = np.min(dataFlatten1)
    band1_max = np.max(dataFlatten1)
    
    data2 = dfTrain['band_2'].as_matrix()
    data2 = np.array(map(lambda x: np.array(x), data2))
    dataFlatten2 = data2.flatten()

    band2_min = np.min(dataFlatten2)
    band2_max = np.max(dataFlatten2)
    print('Train - band_1 => Min:', band1_min, ', Max:', band1_max)
    print('Train - band_2 => Min:', band2_min, ', Max:', band2_max)
    defaultInclination = dfTrain[dfTrain['inc_angle'] != 'na']['inc_angle'].astype(float).mean()

    if dataset == 'train':
        data1 = np.array(map(lambda x: np.histogram(x, bins, (band1_min, band1_max), density=density)[0], data1))
        data2 = np.array(map(lambda x: np.histogram(x, bins, (band2_min, band2_max), density=density)[0], data2))
        df1_ = pd.DataFrame(data1, columns=['band1_r'+str(x) for x in range(bins)])
        df2_ = pd.DataFrame(data2, columns=['band2_r'+str(x) for x in range(bins)])
        dfTrain = pd.concat([dfTrain, df1_, df2_], axis=1)

        dfTrain[["inc_angle"]] = dfTrain[["inc_angle"]].replace('na', defaultInclination)
        dfTrain["inc_angle"] = dfTrain["inc_angle"].astype(float)
        return dfTrain

    dfTest = pd.read_json(testFile)
    data1 = dfTest['band_1'].as_matrix()
    data1 = np.array(map(lambda x: np.array(x), data1))
    data2 = dfTest['band_2'].as_matrix()
    data2 = np.array(map(lambda x: np.array(x), data2))

    data1[data1 > band1_max] = band1_max
    data1[data1 < band1_min] = band1_min
    data2[data2 > band2_max] = band2_max
    data2[data2 < band2_min] = band2_min

    data1 = np.array(map(lambda x: np.histogram(x, bins, (band1_min, band1_max), density=density)[0], data1))
    data2 = np.array(map(lambda x: np.histogram(x, bins, (band2_min, band2_max), density=density)[0], data2))
    df1_ = pd.DataFrame(data1, columns=['band1_r'+str(x) for x in range(bins)])
    df2_ = pd.DataFrame(data2, columns=['band2_r'+str(x) for x in range(bins)])
    dfTest = pd.concat([dfTest, df1_, df2_], axis=1)
    dfTest[["inc_angle"]] = dfTest[["inc_angle"]].replace('na', defaultInclination)
    dfTest["inc_angle"] = dfTest["inc_angle"].astype(float)
    return dfTest

def _generateWindowsBins(x, flip):
    #print(x)
    img1 = np.array(x['band_1']).reshape((75, 75))  
    img2 = np.array(x['band_2']).reshape((75, 75))  
    img3 = img1 + img2
    img3 -= img3.min()
    img3 /= img3.max()
    img3 *= 255

    img = img3.astype(np.uint8)
    img[img<170] = 0

    if flip == 1:
        img = cv2.flip(img, 1)
    elif flip == 0:
        img = cv2.flip(img, 0)
    W = 5
    window_shape = (W, W)
    dict_ = {}

    B = view_as_windows(img, window_shape, step=5)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            matrix = B[i, j]
            data = np.sum(matrix)
            dict_['window_img:' + str(i) + ":" + str(j)] = data

#     img1 = img1 + 50
#     img2 = img2 + 50
# 
#     B = view_as_windows(img1, window_shape, step=5)
#     for i in range(B.shape[0]):
#         for j in range(B.shape[1]):
#             matrix = B[i, j]
#             data = np.sum(matrix)
#             dict_['window_img1:' + str(i) + ":" + str(j)] = data
# 
#     B = view_as_windows(img2, window_shape, step=5)
#     for i in range(B.shape[0]):
#         for j in range(B.shape[1]):
#             matrix = B[i, j]
#             data = np.sum(matrix)
#             dict_['window_img2:' + str(i) + ":" + str(j)] = data
    return dict_

def generateWindowsBins(jsonFile, flip=None): 
    dfTrain = pd.read_json(jsonFile)
    df = dfTrain.apply(_generateWindowsBins, args = (flip,), axis=1)
    df = pd.DataFrame(df.tolist())
    return df

def _generateStructuralFeatures(x):
    #print(x)
    img1 = np.array(x['band_1']).reshape((75, 75))  
    img2 = np.array(x['band_2']).reshape((75, 75))  
    img3 = img1 + img2
    img3 -= img3.min()
    img3 /= img3.max()
    img3 *= 255
    img = img3.astype(np.uint8)
    img[img<170] = 0
    
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = reduce(lambda x, y: x if cv2.contourArea(x) > cv2.contourArea(y) else y, contours)
    area = cv2.contourArea(cnt)
    arcLength = cv2.arcLength(cnt, True)
    x1, y1, width, height = cv2.boundingRect(cnt)    
    rect_area = width * height
    extent = float(area)/rect_area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    band1_min = np.min(np.array(x['band_1']))
    band1_max = np.max(np.array(x['band_1']))
    band1_mean = np.mean(np.array(x['band_1']))
    band1_std = np.std(np.array(x['band_1']))
    band1_var = np.var(np.array(x['band_1']))
    band1_diff = band1_max - band1_min
    band2_min = np.min(np.array(x['band_1']))
    band2_max = np.max(np.array(x['band_2']))
    band2_mean = np.mean(np.array(x['band_2']))
    band2_std = np.std(np.array(x['band_2']))
    band2_var = np.var(np.array(x['band_2']))    
    band2_diff = band2_max - band2_min

    #print(band1_min,band1_max, band1_mean, band1_std, band1_var, band1_diff)
    #exit()
    dict_ = {
        'area': area,
        'arcLength' : arcLength,
        'width': width,
        'height': height,
        'extent': extent,
        'solidity': solidity,
        'band1_min': band1_min,
        'band1_max': band1_max, 
        'band1_mean': band1_mean, 
        'band1_std': band1_std, 
        'band1_var': band1_var, 
        'band1_diff': band1_diff,
        'band2_min': band2_min,
        'band2_max': band2_max, 
        'band2_mean': band2_mean, 
        'band2_std': band2_std, 
        'band2_var': band2_var, 
        'band2_diff': band2_diff,        
    } 
    return dict_

def getStructuralFeatures(jsonFile): 
    dfTrain = pd.read_json(jsonFile)
    df = dfTrain.apply(_generateStructuralFeatures, axis=1)
    df = pd.DataFrame(df.tolist())
    return df

if __name__ == "__main__":
    trainFile = os.path.join("..", "data", "train.json")
    testFile = os.path.join("..", "data", "test.json")
    df = getHistogramsBins(trainFile, testFile, dataset='train')
    print("Train Dataset:", df.shape)
    #df = exploreData(trainFile, testFile, dataset='test')
    #print("Test Dataset:", df.shape)
