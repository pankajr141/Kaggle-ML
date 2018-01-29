'''
Created on Jan 4, 2018

@author: 703188429
'''
import os
import cv2
import numpy as np
import copy

traiImgDir0 = r"../../data/input/train/0"
traiImgDir1 = r"../../data/input/train/1"
imgDir = traiImgDir0
for file_ in os.listdir(imgDir):
    filePath = os.path.join(imgDir, file_)
    print(filePath)
    img = cv2.imread(filePath)
    imgOrig = copy.deepcopy(img)
    img[img<170] = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((3,3),np.uint8)
    #gray = cv2.dilate(gray, kernel, iterations = 1)

    _, contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = reduce(lambda x, y: x if cv2.contourArea(x) > cv2.contourArea(y) else y, contours)
    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])[0]
    extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])[0]
    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])[1]
    extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])[1]
    print(cv2.contourArea(cnt), cv2.arcLength(cnt, True), extLeft, extRight, extTop, extBot)
    cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    x,y,w,h = cv2.boundingRect(cnt)
    print(x,y,w,h)
    
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    print(extent)
    
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    print(solidity)

    """    
    for cnt in contours:
        print cnt
        cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    """
    print(img.shape)
    #cv2.startWindowThread()
    #cv2.namedWindow("preview")
    
    cv2.imshow("preview", np.hstack([imgOrig, img]))
    cv2.waitKey(0)
