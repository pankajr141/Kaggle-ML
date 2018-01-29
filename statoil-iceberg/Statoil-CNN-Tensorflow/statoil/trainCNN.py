'''
Created on Dec 14, 2017

@author: 703188429
'''
import os
import cv2
from sklearn import cross_validation
from sklearn.metrics.classification import confusion_matrix
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from functools import reduce
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def getTrainHoldoutSplit(df, dfLabel, holdoutSize=0.15):
    dfHoldOut = None
    dfHoldOutLabel = None
    cv_pre = cross_validation.StratifiedShuffleSplit(dfLabel, 1, test_size=holdoutSize, random_state=0)
    for train_index, test_index in cv_pre:
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print(train_index.max(), test_index.max())
        print(df.shape, type(df))
        y_train, y_test = dfLabel[train_index], dfLabel[test_index]
        x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        df, dfLabel = x_train, y_train
        dfHoldOut, dfHoldOutLabel = x_test, y_test
    print("==================== Data Set ==================================")
    print("Holdout Set => ", dfHoldOut.shape, dfHoldOutLabel.value_counts())
    print("Train Set => ", df.shape, dfLabel.value_counts())
    print("==================== Data Set ==================================")
    return(df, dfLabel, dfHoldOut, dfHoldOutLabel)


def _generateStructuralFeatures(x):
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
    minArea = 0

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
    band2_min = np.min(np.array(x['band_2']))
    band2_max = np.max(np.array(x['band_2']))
    band2_mean = np.mean(np.array(x['band_2']))
    band2_std = np.std(np.array(x['band_2']))
    band2_var = np.var(np.array(x['band_2']))    
    band2_diff = band2_max - band2_min
    
    ''' Calculated in conventional model part '''
    bins = 30
    density = False
    band1_min_combined, band1_max_combined = -45.594448, 34.574917
    band2_min_combined, band2_max_combined = -45.655499 , 20.154249
    
    data1 = np.histogram(x['band_1'], bins, (band1_min_combined, band1_max_combined), density=density)[0]
    data2 = np.histogram(x['band_2'], bins, (band2_min_combined, band2_max_combined), density=density)[0]
    d1 = {'band1_r'+str(cntr): x for cntr, x in enumerate(data1)}
    d2 = {'band2_r'+str(cntr): x for cntr, x in enumerate(data2)}
    #df1_ = pd.DataFrame(data1, columns=['band1_r'+str(x) for x in range(bins)])
    #df2_ = pd.DataFrame(data2, columns=['band2_r'+str(x) for x in range(bins)])
    #print(df1_)

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
        'inc_angle': x['inc_angle']
    } 
    dict_.update(d1)
    dict_.update(d2)

    return dict_

''' 
Function is dual purpose
1. Split the data into training and test set
2. Feature Engineering
'''
def convertData(df, split=True):

    ''' Convert data into 3D image '''
    #df['band_1'] = df['band_1'].astype(np.float32)
    #df['band_2'] = df['band_2'].astype(np.float32)
    #df = df.copy()

    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    
    
    x2_band1 =  [np.array(band) for band in df["band_1"]]
    x2_band1 = np.array([np.array(band + abs(band.min()) / np.max(band + abs(band.min()))).reshape(75, 75)
                  for band in x2_band1])
    x2_band2 =  [np.array(band) for band in df["band_2"]]
    x2_band2 = np.array([np.array(band + abs(band.min()) / np.max(band + abs(band.min()))).reshape(75, 75)
                  for band in x2_band2])
    
    def _makeImage(x):
        img1, img2 = np.array(x['band_1']), np.array(x['band_2'])
        img3 = img1 + img2
        img3 = ((img3 - img3.min()) / img3.max()) * 255
        img3 = img3.astype(np.uint8)
        return img3

    x3_band3 = x_band1 + x_band2
    x3_band3 = [np.array(x).reshape(75, 75) for x in x3_band3.tolist()]
    x3_band3 = np.array([((x - x.min()) / x.max()) * 255 for x in x3_band3])
    print(x_band1.shape, x_band2.shape)
    print(x2_band1.shape, x2_band2.shape)
    print(x3_band3.shape)

    ''' Image data for convolution Layers - CNN, 
    thus forming a 2 3D images x1, x2 which will evaluated through 2 different CNN '''
    x1 = np.concatenate([
                            x_band1[:, :, :, np.newaxis], 
                            x_band2[:, :, :, np.newaxis],
                            ((x_band1 + x_band2)/2)[:, :, :, np.newaxis],
                            ], axis=-1)
    x2 = np.concatenate([
                            #(x_band1 / x_band2)[:, :, :, np.newaxis],
                            x2_band1[:,:,:,np.newaxis],
                            x2_band2[:,:,:,np.newaxis],
                            x3_band3[:,:,:,np.newaxis]
                            ], axis=-1)

    df['inc_angle'] = df['inc_angle'].astype(str)
    defaultInclination = df[df['inc_angle'] != 'na']['inc_angle'].astype(float).mean()
    defaultInclination = defaultInclination if split else 39.2687074779 # For test data use train default mean of train

    df[['inc_angle']] = df[['inc_angle']].replace('na', defaultInclination)
    df['inc_angle'] = df['inc_angle'].astype(float)
    #inc_angle = df['inc_angle'].tolist()
    
    df = df.reset_index()    
    dfFeatures = df.apply(_generateStructuralFeatures, axis=1)
    dfFeatures = pd.DataFrame(dfFeatures.tolist())
    print("x1 shape:", x1.shape, "x2 shape:", x2.shape)
    print("dfFeatures shape:", dfFeatures.shape)

    ''' convert all features into list so that it cab be easily passed to tensor flow'''
    x1 = x1.tolist()
    x2 = x2.tolist()
    xFeatures = dfFeatures.as_matrix().tolist()
    
    '''
    img - input for 1st CNN
    img2 - input for 2nd CNN
    features are added to final Fully connected layer
    '''
    df['img'] = pd.Series(x1, dtype=np.dtype("object"))
    df['img2'] = pd.Series(x2, dtype=np.dtype("object"))
    df['features'] = pd.Series(xFeatures, dtype=np.dtype("object"))
    
    ''' Check whether Train or Test for Train split = True'''
    if not split:
        return df['id'], df[['img', 'img2', 'features']]  # Return Test data

    dfLabel = df['is_iceberg']
    #df.drop(['is_iceberg'], axis=1, inplace=True)
    df, dfLabel, dfHoldOut, dfHoldOutLabel = getTrainHoldoutSplit(df, dfLabel, holdoutSize=0.20)
    #train, test = train_test_split(df)
    #print(train.shape, test.shape)
    #print(df.shape, dfLabel.shape, dfHoldOut.shape, dfHoldOutLabel.shape)
    '''
    trainX = train['img'].tolist()
    trainX = map(lambda x: np.array(ast.literal_eval(x)).reshape((75, 75)), trainX)
    trainXAngle = train['inc_angle'].tolist()
    trainXIsIceberg = train['is_iceberg'].tolist()
    '''
    return df[['img', 'img2', 'features']], dfLabel, dfHoldOut[['img', 'img2', 'features']], dfHoldOutLabel


''' Entry Point'''
def main(unused_argv):
    TRAIN = True  # set True for Training and set False for prediction
    from datetime import datetime
    starttime = datetime.now()

    import model  # Contain the model file or CNN model arch used
    cnn_classifier = tf.estimator.Estimator(model_fn=model.cnn_model_fn, model_dir="tmp/statoil_convnet_model")

    ''' This will only be executed during training '''
    if TRAIN == True:
        jsonFile = os.path.join("data", "train.json")
        df = pd.read_json(jsonFile)
        df = df.sort_values('band_1') # TO make data 
        
        ''' Getting features and holdout set using function: convertData'''
        trainX, trainY, testX, testY =  convertData(df)

        trainX1 = np.array(trainX['img'].tolist())
        trainX2 = np.array(trainX['img2'].tolist())
        trainFeatures = np.array(trainX['features'].tolist())

        testX1 = np.array(testX['img'].tolist())
        testX2 = np.array(testX['img2'].tolist())
        testFeatures = np.array(testX['features'].tolist())

        train_labels = np.array(trainY.tolist())
        test_labels = np.array(testY.tolist())

        print("Shapes:", trainX1.shape, trainX2.shape, trainFeatures.shape, train_labels.shape)
        
        '''Adding some randomness by flipping images and adding as additional datasets
        Note that every thing else will remain same only orientation of image will change
        '''
        def _flipH(x):
            return np.flip(x, -1)
        def _flipV(x):
            return np.flip(x, -1)        

        trainX1_flipH = np.array(list(map(lambda x: _flipH(x), trainX1)))
        trainX1_flipV = np.array(list(map(lambda x: _flipV(x), trainX1)))
        trainX2_flipH = np.array(list(map(lambda x: _flipH(x), trainX2)))
        trainX2_flipV = np.array(list(map(lambda x: _flipV(x), trainX2)))

        trainX1 = np.concatenate([trainX1, trainX1_flipH, trainX1_flipV], axis=0)
        trainX2 = np.concatenate([trainX2, trainX2_flipH, trainX2_flipV], axis=0)
        trainFeatures = np.concatenate([trainFeatures, trainFeatures, trainFeatures], axis=0)
        train_labels = np.concatenate([train_labels, train_labels, train_labels], axis=0)

        print("Shapes After Flipping:", trainX1.shape, trainX2.shape, trainFeatures.shape, train_labels.shape)


        ''' Adding Estimator + Model params etc, below is related to tensorflow'''

        ''' Adding hooks and monitors to display during training phase '''    
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=20)
        #validation_monitor = tf.contrib.learn.monitors.ValidationMonitor({"x": testX}, test_labels, every_n_steps=100)
        #validation_monitor = tf.contrib.learn.monitors.ValidationMonitor({"x": testX}, test_labels, 
        #                                                                 every_n_steps=100)

        #list_of_monitors_and_hooks = [ validation_monitor,
                                        #logging_hook
        #                            ]
        #hooks = monitor_lib.replace_monitors_with_hooks(list_of_monitors_and_hooks, cnn_classifier)
        #print ("Hooks:", hooks)

        '''Train - dataset and steps + additonal parameter define, If any tuning needs to be made for make here
        like batch_size. reduce batch_size if memory overflow occuring'''
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x1": trainX1, 'x2': trainX2, 'features': trainFeatures}, 
                                                            y=train_labels, 
                                                            batch_size=25, 
                                                            num_epochs=None, 
                                                            shuffle=True)

        '''Evaluate the model and print results'''
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x1": testX1, 'x2': testX2, 'features': testFeatures},
                                                           y=test_labels,
                                                           num_epochs=1,
                                                           shuffle=False)
        
        
        '''Below commented code should be used if not using experiment interface, experiment is used because we needed
        to see the validation score after some intervals.''' 
        
        '''
        cnn_classifier.train(input_fn=train_input_fn, steps=1000,
                             hooks=hooks
                             )
        eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        '''

        #eval_metrics={'accuracy': MetricSpec(tfmetrics.streaming_accuracy)}
        experiment = tf.contrib.learn.Experiment(
                                            estimator=cnn_classifier,
                                            train_input_fn=train_input_fn,
                                            eval_input_fn=eval_input_fn,
                                            #eval_metrics=['accuracy']
                                            train_steps=16000,
                                            min_eval_frequency=1
        )
        experiment.train_and_evaluate()

        '''
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=3000)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, start_delay_secs=0)
        tf.estimator.train_and_evaluate(cnn_classifier, train_spec, eval_spec)
        '''

        ''' Predict on holdout set print results'''
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x1": testX1, 'x2': testX2, 'features': testFeatures},
            num_epochs=1,
            shuffle=False)

        predict_results = cnn_classifier.predict(input_fn=predict_input_fn)
        df = pd.DataFrame(predict_results)
        print(confusion_matrix(test_labels, df['classes'].tolist()))
        print("Training Completed")
        return

   
    ''' Predicting and submitting submission file, will be called if Train=False '''
   
    print("Predicting and submitting submission file")
    jsonFile = os.path.join("data", "test_m.json")
    submissionLst = []
    for chunk in pd.read_json(jsonFile, chunksize=1000, lines=True):  
        ids, testData = convertData(chunk, split=False)
        print("Chunk Shape:", testData.shape)
        #print(testData)
        testX1 = np.array(testData['img'].tolist())
        testX2 = np.array(testData['img2'].tolist())
        testFeatures = np.array(testData['features'].tolist())

        ids = np.array(ids.tolist())
        # Predict and print results
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x1": testX1, 'x2': testX2, 'features': testFeatures},
            num_epochs=1,
            shuffle=False)

        predict_results = cnn_classifier.predict(input_fn=predict_input_fn)
        df = pd.DataFrame(predict_results)
        df['is_iceberg'] = df['probabilities'].apply(lambda x: x[1])
        df['is_iceberg'] = df['is_iceberg'].astype(np.float32)
        is_iceberg = df['is_iceberg'].tolist()
        #pd.set_option('display.float_format', lambda x: '%.99f' % x)
        submission = pd.DataFrame()
        submission['id'] = pd.Series(ids)
        submission['is_iceberg'] = pd.Series(is_iceberg)
        print(">>>", submission.shape)
        submissionLst.append(submission)

    submissionDF = pd.concat(submissionLst, axis=0)
    submissionDF.to_csv('sub.csv', index=False)
    print(datetime.now() - starttime)

if __name__ == "__main__":
    tf.app.run()
