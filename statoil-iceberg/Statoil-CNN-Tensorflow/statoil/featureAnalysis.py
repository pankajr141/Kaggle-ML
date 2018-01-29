'''
Created on 24-Dec-2017

@author: amuse
'''

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

trainJson = r"data\train.json"
dfO = pd.read_json(trainJson)
df = dfO[dfO['is_iceberg'] == 0]
df['band_1'] = df['band_1'].apply(lambda x: np.array(x))
print(df['band_1'].describe())

x = 0
fig, ax = plt.subplots(figsize=(10,5), ncols=4, nrows=2)
seaborn.distplot(df['band_1'].tolist()[x], ax=ax[0][0])
seaborn.distplot(df['band_1'].tolist()[x+1], ax=ax[0][1])
seaborn.distplot(df['band_1'].tolist()[x+2], ax=ax[0][2])
seaborn.distplot(df['band_1'].tolist()[x+3], ax=ax[0][3])

df = dfO[dfO['is_iceberg'] == 1]
x = 0
seaborn.distplot(df['band_1'].tolist()[x], ax=ax[1][0])
seaborn.distplot(df['band_1'].tolist()[x+1], ax=ax[1][1])
seaborn.distplot(df['band_1'].tolist()[x+2], ax=ax[1][2])
seaborn.distplot(df['band_1'].tolist()[x+3], ax=ax[1][3])
plt.show()

a = range(100,300,2)
print(list(a) + list(a))
print(np.median(a))