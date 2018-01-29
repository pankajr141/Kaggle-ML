'''
Created on Dec 8, 2017

@author: 703188429
'''
import os
import pandas as pd
import math
import numpy as np
#os.makedirs("data/input/train/1")
#os.makedirs("data/input/train/0")

def convertDBImagesToJpgAndCsv(writeImage=False):
    trainFile = os.path.join("data", "train.json")
    df_train = pd.read_json(trainFile)
    lst = []
    for ix, row in df_train.iterrows():
        img1 = np.array(row['band_1']).reshape((75, 75))  
        img2 = np.array(row['band_2']).reshape((75, 75))
        inc_angle = row['inc_angle']
        is_iceberg = row['is_iceberg']
        img3 = img1 + img2
        img3 -= img3.min()
        img3 /= img3.max()
        img3 *= 255
        img3 = img3.astype(np.uint8)
        if row['is_iceberg']==0:
            if writeImage:
                cv2.imwrite("data/input/train/0/f{}.png".format(ix), img3)
        elif row['is_iceberg']==1:
            if writeImage:
                cv2.imwrite("data/input/train/1/f{}.png".format(ix), img3)
        lst.append({
            'img': img3.ravel().tolist(),
            'is_iceberg': is_iceberg,
            'inc_angle': inc_angle
        })
    pd.DataFrame(lst).to_csv("data/train.csv", index=False)
if __name__ == "__main__":
    convertDBImagesToJpgAndCsv(writeImage=False)
# print df.head(4)
# v = 10
# band_1 = df['band_1'].tolist()[1:v]
# img  = np.array(band_1).reshape((v-1, 75, 75))
# print img
