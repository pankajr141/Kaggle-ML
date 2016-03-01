'''
Created on 18-Jan-2016

@author: pankajrawat
'''
import bisect
import os
from sklearn.externals import joblib
import numpy as np
import pandas as pd


event_path = os.path.join('inputs', 'event_type.csv')
logfeature_path = os.path.join('inputs', 'log_feature.csv')
resourcetype_path = os.path.join('inputs', 'resource_type.csv')
severitytype_path = os.path.join('inputs', 'severity_type.csv')
train_path = os.path.join('inputs', 'train.csv')
test_path = os.path.join('inputs', 'test.csv')

df_pickle_path = os.path.join('objects', 'df.pickle')
dfLabel_pickle_path = os.path.join('objects', 'dfLabel.pickle')
dfTest_pickle_path = os.path.join('objects', 'dfTest.pickle')


def get_logfeature_df():
    logfeature_df = pd.read_csv(logfeature_path)
    logfeature_df.rename(columns=lambda x: x.replace('log_feature', 'lf'), inplace=True)
    logfeature_df['lf'] = logfeature_df['lf'].str.replace('feature ', '')
    x = logfeature_df[['lf']]
    x = pd.get_dummies(x)
    logfeature_df = pd.concat([logfeature_df, x], axis=1)
    for column in logfeature_df.columns:
        if not column.startswith('lf_'):
            continue
        logfeature_df[column] = logfeature_df[column] * logfeature_df['volume']
    logfeature_df.drop(['lf', 'volume'], axis=1, inplace=True)
    logfeature_df = logfeature_df.groupby(['id'], as_index=False).sum()
    return logfeature_df


def get_eventtype_df():
    event_df = pd.read_csv(event_path)
    event_df.rename(columns=lambda x: x.replace('event_type', 'et'), inplace=True)
    event_df['et'] = event_df['et'].str.replace('event_type ', '')
    x = event_df[['et']]
    x = pd.get_dummies(x)
    event_df = pd.concat([event_df, x], axis=1)
    event_df.drop(['et'], axis=1, inplace=True)
    event_df = event_df.groupby(['id'], as_index=False).sum()
    return event_df


def get_resourcetype_df():
    resourcetype_df = pd.read_csv(resourcetype_path)
    resourcetype_df.rename(columns=lambda x: x.replace('resource_type', 'rt'), inplace=True)
    resourcetype_df['rt'] = resourcetype_df['rt'].str.replace('resource_type ', '')
    x = resourcetype_df[['rt']]
    x = pd.get_dummies(x)
    resourcetype_df = pd.concat([resourcetype_df, x], axis=1)
    resourcetype_df.drop(['rt'], axis=1, inplace=True)
    resourcetype_df = resourcetype_df.groupby(['id'], as_index=False).sum()
    return resourcetype_df


def get_severitytype_df():
    severitytype_df = pd.read_csv(severitytype_path)
    severitytype_df.rename(columns=lambda x: x.replace('severity_type', 'st'), inplace=True)
    severitytype_df['st'] = severitytype_df['st'].str.replace('severity_type ', '')
    x = severitytype_df[['st']]
    x = pd.get_dummies(x)
    severitytype_df = pd.concat([severitytype_df, x], axis=1)
    severitytype_df.drop(['st'], axis=1, inplace=True)
    severitytype_df = severitytype_df.groupby(['id'], as_index=False).sum()
    return severitytype_df


def calculate_custom_features(Maindf, path):
    # Severity Impact
    stColumns = filter(lambda x: x.startswith('st_'), Maindf.columns)
    len_df = len(Maindf)
    #stColumns = map(lambda x: int(x.replace('st_', '')), stColumns)
    global st_impact_on_severity
    if path == train_path:
        st_impact_on_severity = {}
        for stColumn in stColumns:
            tmpdf = Maindf[Maindf[stColumn] > 0]
            framelen = tmpdf.shape[0]
            impact_on_0 = tmpdf[tmpdf['fault_severity'] == 0].shape[0] / float(framelen)
            impact_on_1 = tmpdf[tmpdf['fault_severity'] == 1].shape[0] / float(framelen)
            impact_on_2 = tmpdf[tmpdf['fault_severity'] == 2].shape[0] / float(framelen)

            impact_avg_0 = tmpdf[tmpdf['fault_severity'] == 0].shape[0] / float(Maindf[Maindf['fault_severity'] == 0].shape[0])
            impact_avg_1 = tmpdf[tmpdf['fault_severity'] == 1].shape[0] / float(Maindf[Maindf['fault_severity'] == 1].shape[0])
            impact_avg_2 = tmpdf[tmpdf['fault_severity'] == 2].shape[0] / float(Maindf[Maindf['fault_severity'] == 2].shape[0])

            st_impact_on_severity[stColumn] = [impact_on_0, impact_on_1, impact_on_2, impact_avg_0, impact_avg_1, impact_avg_2]
    imapctDF = pd.DataFrame(np.zeros(shape=(len_df, 6)), columns=['stI_0', 'stI_1', 'stI_2', 'stA_0', 'stA_1', 'stA_2'])
    Maindf = pd.concat([Maindf, imapctDF], axis=1)

    def fStImpact(x, label):
        for stColumn in stColumns:
            if x[stColumn] == 1:
                return st_impact_on_severity[stColumn][label]

    def fStAverage(x, label):
        for stColumn in stColumns:
            if x[stColumn] == 1:
                return st_impact_on_severity[stColumn][label + 3]

    def fstReaches2(x):
        if x['st_1'] == 1 or x['st_2'] == 1:
            return 1
        return 0

    # For this Severity type what is the percentage that fault_severity would be 0
    Maindf['stI_0'] = Maindf.apply(fStImpact, args=[0], axis=1)
    Maindf['stI_1'] = Maindf.apply(fStImpact, args=[1], axis=1)
    Maindf['stI_2'] = Maindf.apply(fStImpact, args=[2], axis=1)

    # For this Severity type, Average among all severity types on fault_serverity 0
    Maindf['stA_0'] = Maindf.apply(fStAverage, args=[0], axis=1)
    Maindf['stA_1'] = Maindf.apply(fStAverage, args=[1], axis=1)
    Maindf['stA_2'] = Maindf.apply(fStAverage, args=[2], axis=1)

    # For this Severity type does impact ever reaches 2
    Maindf['stC_2'] = Maindf.apply(fstReaches2, axis=1)

    # Event Impact
    etColumns = filter(lambda x: x.startswith('et_'), Maindf.columns)
    global et_impact_on_severity
    if path == train_path:
        et_impact_on_severity = {}
        for etColumn in etColumns:
            tmpdf = Maindf[Maindf[etColumn] > 0]
            framelen = tmpdf.shape[0]
            impact_on_0, impact_on_1, impact_on_2 = 0, 0, 0
            if framelen:
                impact_on_0 = tmpdf[tmpdf['fault_severity'] == 0].shape[0] / float(framelen)
                impact_on_1 = tmpdf[tmpdf['fault_severity'] == 1].shape[0] / float(framelen)
                impact_on_2 = tmpdf[tmpdf['fault_severity'] == 2].shape[0] / float(framelen)
            et_impact_on_severity[etColumn] = [impact_on_0, impact_on_1, impact_on_2]

    imapctDF = pd.DataFrame(np.zeros(shape=(len_df, 3)), columns=['etI_0', 'etI_1', 'etI_2'])
    Maindf = pd.concat([Maindf, imapctDF], axis=1)

    def fEtImpact(x, label):
        impact = 0.0
        for etColumn in etColumns:
            if x[etColumn] == 1:
                impact += et_impact_on_severity[etColumn][label]
        return impact

    Maindf['etI_0'] = Maindf.apply(fEtImpact, args=[0], axis=1)
    Maindf['etI_1'] = Maindf.apply(fEtImpact, args=[1], axis=1)
    Maindf['etI_2'] = Maindf.apply(fEtImpact, args=[2], axis=1)

    # Resource Impact
    rtColumns = filter(lambda x: x.startswith('rt_'), Maindf.columns)
    global rt_impact_on_severity
    if path == train_path:
        rt_impact_on_severity = {}
        for rtColumn in rtColumns:
            tmpdf = Maindf[Maindf[rtColumn] > 0]
            framelen = tmpdf.shape[0]
            impact_on_0, impact_on_1, impact_on_2 = 0, 0, 0
            if framelen:
                impact_on_0 = tmpdf[tmpdf['fault_severity'] == 0].shape[0] / float(framelen)
                impact_on_1 = tmpdf[tmpdf['fault_severity'] == 1].shape[0] / float(framelen)
                impact_on_2 = tmpdf[tmpdf['fault_severity'] == 2].shape[0] / float(framelen)
            rt_impact_on_severity[rtColumn] = [impact_on_0, impact_on_1, impact_on_2]

    imapctDF = pd.DataFrame(np.zeros(shape=(len_df, 3)), columns=['rtI_0', 'rtI_1', 'rtI_2'])
    Maindf = pd.concat([Maindf, imapctDF], axis=1)

    def fRtImpact(x, label):
        impact = 0.0
        for rtColumn in rtColumns:
            if x[rtColumn] == 1:
                impact += rt_impact_on_severity[rtColumn][label]
        return impact

    Maindf['rtI_0'] = Maindf.apply(fRtImpact, args=[0], axis=1)
    Maindf['rtI_1'] = Maindf.apply(fRtImpact, args=[1], axis=1)
    Maindf['rtI_2'] = Maindf.apply(fRtImpact, args=[2], axis=1)

    # Location Impact
    ltColumns = filter(lambda x: x.startswith('loc_'), Maindf.columns)
    global lt_impact_on_severity
    if path == train_path:
        lt_impact_on_severity = {}
        for ltColumn in ltColumns:
            tmpdf = Maindf[Maindf[ltColumn] > 0]
            framelen = tmpdf.shape[0]
            impact_on_0, impact_on_1, impact_on_2 = 0, 0, 0
            if framelen:
                impact_on_0 = tmpdf[tmpdf['fault_severity'] == 0].shape[0] / float(framelen)
                impact_on_1 = tmpdf[tmpdf['fault_severity'] == 1].shape[0] / float(framelen)
                impact_on_2 = tmpdf[tmpdf['fault_severity'] == 2].shape[0] / float(framelen)
            lt_impact_on_severity[ltColumn] = [impact_on_0, impact_on_1, impact_on_2]

    imapctDF = pd.DataFrame(np.zeros(shape=(len_df, 3)), columns=['locI_0', 'locI_1', 'locI_2'])
    Maindf = pd.concat([Maindf, imapctDF], axis=1)

    # create average impact for location which are not present in train.csv
    lt_keys = sorted(map(lambda x: int(x.replace('loc_', '')), lt_impact_on_severity.keys()))

    def fLtImpact(x, label):
        known_locations = lt_impact_on_severity.keys()  # These are the location provided in train.csv
        for ltColumn in ltColumns:
            if x[ltColumn] == 1:
                if not ltColumn in known_locations:     # Their are some locations not mentioned in test.csv so they would have a zero impact
                    ltKey = int(ltColumn.split('_')[1])
                    index = bisect.bisect_left(lt_keys, ltKey)

                    keys_to_consider = 2
                    left_lt_keys = map(lambda x: "loc_" + str(x), lt_keys[index - keys_to_consider: index])
                    right_lt_keys = map(lambda x: "loc_" + str(x), lt_keys[index: index + keys_to_consider])
                    avg_impact_label = 0.0
                    for left_lt_key in left_lt_keys:
                        avg_impact_label += lt_impact_on_severity[left_lt_key][label]
                    for right_lt_key in right_lt_keys:
                        avg_impact_label += lt_impact_on_severity[right_lt_key][label]
                    avg_impact_label = float(avg_impact_label) / len(left_lt_keys + right_lt_keys)
                    #print ltColumn, " : ", label, " => ", avg_impact_label
                    return avg_impact_label
                return lt_impact_on_severity[ltColumn][label]
        #return impact

    Maindf['locI_0'] = Maindf.apply(fLtImpact, args=[0], axis=1)
    Maindf['locI_1'] = Maindf.apply(fLtImpact, args=[1], axis=1)
    Maindf['locI_2'] = Maindf.apply(fLtImpact, args=[2], axis=1)

#===============================================================================
#     for key, value in lt_impact_on_severity.items():
#         nkey = int(key.replace('loc_', ''))
#         lt_impact_on_severity.pop(key, None)
#         lt_impact_on_severity[nkey] = value
#
#     for key, value in sorted(lt_impact_on_severity.items()):
#         print key, '\t', value
#===============================================================================
    return Maindf

a = pd.read_csv(train_path)
b = pd.read_csv(test_path)
extra_loc_train = pd.Series(list(set(a['location']) - set(b['location'])))
extra_loc_test = pd.Series(list(set(b['location']) - set(a['location'])))
extra_loc_train = extra_loc_train.str.replace('location ', 'loc_')
extra_loc_test = extra_loc_test.str.replace('location ', 'loc_')

#print len(extra_loc_test), extra_loc_test


def get_final_df(path):
    logfeature_df = get_logfeature_df()
    eventtype_df = get_eventtype_df()
    resourcetype_df = get_resourcetype_df()
    severitytype_df = get_severitytype_df()
    df = pd.read_csv(path)
    df.rename(columns=lambda x: x.replace('location', 'loc'), inplace=True)
    df['loc'] = df['loc'].str.replace('location ', '')
    x = df[['loc']]
    x = pd.get_dummies(x)
    df = pd.concat([df, x], axis=1)
    df.drop(['loc'], axis=1, inplace=True)

    df = pd.merge(df, logfeature_df, on=['id'])
    df = pd.merge(df, eventtype_df, on=['id'])
    df = pd.merge(df, resourcetype_df, on=['id'])
    df = pd.merge(df, severitytype_df, on=['id'])
    df = calculate_custom_features(df, path)

    if path == train_path:
        extra_loc_df = pd.DataFrame(np.zeros(shape=(len(df), len(extra_loc_test))), columns=extra_loc_test)
    elif path == test_path:
        extra_loc_df = pd.DataFrame(np.zeros(shape=(len(df), len(extra_loc_train))), columns=extra_loc_train)

    df = pd.concat([df, extra_loc_df], axis=1)
    df = df.reindex_axis(sorted(df.columns), axis=1)
    return df

df = get_final_df(train_path)
dfLabel = df['fault_severity']
df.drop(['fault_severity'], axis=1, inplace=True)

joblib.dump(df, df_pickle_path)
joblib.dump(dfLabel, dfLabel_pickle_path)

dfTest = get_final_df(test_path)
joblib.dump(dfTest, dfTest_pickle_path)
