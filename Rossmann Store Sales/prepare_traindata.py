'''
Created on 08-Nov-2015

@author: pankajrawat
'''
from datetime import datetime
import math
import os

from sklearn import preprocessing

import numpy as np
import pandas as pd


df = pd.read_csv('train.csv')
df_store = pd.read_csv('store.csv')
df_store = df_store.set_index('Store')

df_sales = df['Sales']
len_df = len(df)
df = df.drop(['Customers', 'Sales'], axis=1)

#storeTypeDF = pd.DataFrame(np.zeros(shape=(len_df, 4)), columns=['st_a', 'st_b', 'st_c', 'st_d'])
#assortmentDF = pd.DataFrame(np.zeros(shape=(len_df, 3)), columns=['asmt_a', 'asmt_b', 'asmt_c'])
#promo2DF = pd.DataFrame(np.zeros(shape=(len_df, 1)), columns=['Promo2'])
#df = pd.concat([df, assortmentDF, storeTypeDF, promo2DF], axis=1).fillna(0)

fStoreType = lambda x, y: 1.0 if df_store.iloc[int(x) - 1]['StoreType'] == y else 0.0
fAssortment = lambda x, y: 1.0 if df_store.iloc[int(x) - 1]['Assortment'] == y else 0.0
df['Date'] = pd.to_datetime(df['Date'])

df['st_a'] = df['Store'].apply(fStoreType, args=['a'])
df['st_b'] = df['Store'].apply(fStoreType, args=['b'])
df['st_c'] = df['Store'].apply(fStoreType, args=['c'])
df['st_d'] = df['Store'].apply(fStoreType, args=['d'])
df['asmt_a'] = df['Store'].apply(fAssortment, args=['a'])
df['asmt_b'] = df['Store'].apply(fAssortment, args=['b'])
df['asmt_c'] = df['Store'].apply(fAssortment, args=['c'])

df['Christmas'] = df.apply(lambda x: 1.0 if x['StateHoliday'] == 'c' else 0.0, axis=1)
df['StateHoliday'] = df.apply(lambda x: 1.0 if x['StateHoliday'] in ['a', 'b', 'c'] else x['StateHoliday'], axis=1)

df_store['CompetitionOpenSinceMonth'] = df_store['CompetitionOpenSinceMonth']\
                                    .fillna(math.floor(df_store['CompetitionOpenSinceMonth'].mean()))
df_store['CompetitionOpenSinceYear'] = df_store['CompetitionOpenSinceYear']\
                                    .fillna(math.floor(df_store['CompetitionOpenSinceYear'].mean()))

competitionPresentDF = pd.DataFrame(np.zeros(shape=(len_df, 1)), columns=['competitionPresent'])
competitionDistanceDF = pd.DataFrame(np.zeros(shape=(len_df, 1)), columns=['competitionDistance']).replace(0, 999999.9)
df = pd.concat([df, competitionPresentDF, competitionDistanceDF], axis=1)

cntr = 1
def fmodifyCompetition(x):
    storeForThisRow = x['Store']
    dateForThisRow = x['Date']
    global cntr
    cntr += 1
    if cntr % 50000 == 0:
        print "fmodifyCompetition CNTR => ", cntr
    competitionOpenSinceMonth = df_store.iloc[storeForThisRow - 1]['CompetitionOpenSinceMonth']
    competitionOpenSinceYear = df_store.iloc[storeForThisRow - 1]['CompetitionOpenSinceYear']
    dateCompetitionStart = datetime(int(competitionOpenSinceYear), int(competitionOpenSinceMonth), 1)
    if (dateCompetitionStart <= dateForThisRow):
        return df_store.iloc[storeForThisRow - 1]['CompetitionDistance']
    return x['competitionDistance']

df['competitionDistance'] = df.apply(fmodifyCompetition, axis=1)
df['competitionPresent'] = df.apply(lambda x: 0.0 if x['competitionDistance'] > 999990.9 else 1.0, axis=1)
df['competitionDistance'] = df['competitionDistance'].fillna(df['competitionDistance'].mean())
df['competitionDistance'] = preprocessing.MinMaxScaler().fit_transform(df['competitionDistance'])


cntr = 1
def fPromo2(x):
    storeForThisRow = x['Store']
    dateForThisRow = x['Date']
    global cntr
    cntr += 1
    if cntr % 50000 == 0:
        print "fPromo2 CNTR => ", cntr

    # iF store does not participate in promo2 return 0.0
    if df_store.iloc[storeForThisRow - 1]['Promo2'] == 0:
        return 0.0

    promo2SinceWeek = df_store.iloc[storeForThisRow - 1]['Promo2SinceWeek']
    promo2SinceYear = df_store.iloc[storeForThisRow - 1]['Promo2SinceYear']
    dtPromoStart = datetime(int(promo2SinceYear), int(math.ceil((promo2SinceWeek * 7) / 30)), int(math.ceil((promo2SinceWeek * 7) % 30)))    
    # iF date is less then promostartdate return 0.0
    if (dateForThisRow < dtPromoStart):
        return 0.0
    # Return promo
    return x['Promo']

df['promo2'] = df.apply(fPromo2, axis=1)
print df

df.save(os.path.join('objects', 'df.pickle'))
df_sales.save(os.path.join('objects', 'df_sales.pickle'))
