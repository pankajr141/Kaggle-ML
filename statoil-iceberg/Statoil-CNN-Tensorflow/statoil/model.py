'''
Created on 24-Dec-2017

@author: amuse
'''

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"""
Define the model function which include 2CNN Model + Features N/w feeded into a fully DNN model

"""
def cnn_model_fn(features, labels, mode):
    print(mode)

    x1 = features["x1"]
    x1_image = tf.reshape(x1, [-1, 75, 75, 3])  # 1st Argument denotes any number of x in array, 2,3,4 are shape and channels
    x1_image = tf.cast(x1_image, tf.float32)

    x2 = features["x2"]
    x2_image = tf.reshape(x2, [-1, 75, 75, 3])  # 1st Argument denotes any number of x in array, 2,3,4 are shape and channels
    x2_image = tf.cast(x2_image, tf.float32)

    features = features["features"]
    features = tf.cast(features, tf.float32)

    
    '''CNN - 1: For imagre 1 '''
    #b_conv1 = bias_variable([32])
    X1_conv1 = tf.layers.conv2d(inputs=x1_image, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, 
                             #bias_initializer=b_conv1
                             name='X1_conv1')
    #conv1 += b_conv1
    X1_pool1 = tf.layers.max_pooling2d(inputs=X1_conv1, pool_size=[2, 2], strides=2, name='X1_pool1')
    X1_pool1 = tf.layers.dropout(inputs=X1_pool1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    """
    # After h_pool1 the ouptut image has size 38 X 38 for 75x75 pixel image
    out_height = ceil(float(in_height) / float(strides[1]))
    out_width = ceil(float(in_width) / float(strides[2]))
    """

    # Convolutional Layer #2 and Pooling Layer #2
    X1_conv2 = tf.layers.conv2d(inputs=X1_pool1, filters=128, kernel_size=[3, 3], 
                             padding="same", activation=tf.nn.relu, name='X1_conv2')
    X1_pool2 = tf.layers.max_pooling2d(inputs=X1_conv2, pool_size=[2, 2], strides=2, name='X1_pool2')
    X1_pool2 = tf.layers.dropout(inputs=X1_pool2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    X1_conv3 = tf.layers.conv2d(inputs=X1_pool2, filters=64, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu, name='X1_conv3')
    X1_pool3 = tf.layers.max_pooling2d(inputs=X1_conv3, pool_size=[2, 2], strides=2, name='X1_pool3')    
    X1_pool3 = tf.layers.dropout(inputs=X1_pool3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    X1_conv4 = tf.layers.conv2d(inputs=X1_pool3, filters=64, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu, name='X1_conv4')
    X1_pool4 = tf.layers.max_pooling2d(inputs=X1_conv4, pool_size=[2, 2], strides=2, name='X1_pool4')    
    X1_pool4 = tf.layers.dropout(inputs=X1_pool4, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    '''CNN - 2: For imagre 2 '''
    #b_conv1 = bias_variable([32])
    X2_conv1 = tf.layers.conv2d(inputs=x2_image, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, 
                             #bias_initializer=b_conv1
                             name='X2_conv1')
    #conv1 += b_conv1
    X2_pool1 = tf.layers.max_pooling2d(inputs=X2_conv1, pool_size=[2, 2], strides=2, name='X2_pool1')
    X2_pool1 = tf.layers.dropout(inputs=X2_pool1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    """
    # After h_pool1 the ouptut image has size 38 X 38 for 75x75 pixel image
    out_height = ceil(float(in_height) / float(strides[1]))
    out_width = ceil(float(in_width) / float(strides[2]))
    """
    
    # Convolutional Layer #2 and Pooling Layer #2
    X2_conv2 = tf.layers.conv2d(inputs=X2_pool1, filters=128, kernel_size=[3, 3], 
                             padding="same", activation=tf.nn.relu, name='X2_conv2')
    X2_pool2 = tf.layers.max_pooling2d(inputs=X2_conv2, pool_size=[2, 2], strides=2, name='X2_pool2')
    X2_pool2 = tf.layers.dropout(inputs=X2_pool2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    X2_conv3 = tf.layers.conv2d(inputs=X2_pool2, filters=64, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu, name='X2_conv3')
    X2_pool3 = tf.layers.max_pooling2d(inputs=X2_conv3, pool_size=[2, 2], strides=2, name='X2_pool3')    
    X2_pool3 = tf.layers.dropout(inputs=X2_pool3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    X2_conv4 = tf.layers.conv2d(inputs=X2_pool3, filters=64, kernel_size=[3, 3],
                             padding="same", activation=tf.nn.relu, name='X2_conv4')
    X2_pool4 = tf.layers.max_pooling2d(inputs=X2_conv4, pool_size=[2, 2], strides=2, name='X2_pool4')    
    X2_pool4 = tf.layers.dropout(inputs=X2_pool4, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)


    ''' CNN are done, now feed both CNN output + features into DNN model '''
    X1_pool_flat = tf.reshape(X1_pool4, [-1, 4 * 4 * 64])
    X2_pool_flat = tf.reshape(X2_pool4, [-1, 4 * 4 * 64])
    features = tf.reshape(features, [-1, 79])

    pool_flat = tf.concat([X1_pool_flat, X2_pool_flat, features], axis=1)
    print('Tensors shape Before DNN:', X1_pool4, '\n', X2_pool4, '\n', X1_pool_flat, '\n', 
          X2_pool_flat, '\n', pool_flat, '\n', features)

    ''' 3 Layer DNN Model'''
    """ Output Image 9x9 """
    dense = tf.layers.dense(inputs=pool_flat, units=2000, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense = tf.layers.dense(inputs=dropout, units=1400, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense = tf.layers.dense(inputs=dropout, units=1000, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    ''' Softmax layer for final classifications '''
    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
 
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          #eval_metric_ops= eval_metric_ops,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, 
                                      loss=loss, 
                                      eval_metric_ops=eval_metric_ops)
