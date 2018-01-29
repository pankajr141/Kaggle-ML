'''
Created on 22-Dec-2017

@author: amuse
'''
import itertools

import pandas as pd
import ijson
import json
import numpy as np

# for df in pd.read_json(r"data/test_m.json", lines=True, chunksize=100):
#     print(df.shape)
#     print(df.columns)
#     print(type(df['band_1'].tolist()[0]))
#     x_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
# 
# #print(df['band_1'])
# exit()
fp = open("data/test_m.json", 'w')
jsonFile = r"data/test.json"
js  = json.load(open(jsonFile))
print(len(js))
for js_i in js:
    fp.write(json.dumps(js_i) + "\n")
fp.close()