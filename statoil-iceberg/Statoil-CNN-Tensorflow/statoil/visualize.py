'''
Created on Dec 20, 2017

@author: 703188429
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    w = weights

    # Get the lowest and highest values for the weights.This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def visualize(metaFile, checkpoint, latestModel):
    #"""Print All the tensor variable name and data saved in checkpoint file"""
    #print_tensors_in_checkpoint_file(file_name=latestModel, tensor_name='', all_tensors=True)

    """ Accessing Particular variable from checkpoint file """
    with tf.Session() as sess:
        model = tf.train.import_meta_graph(metaFile)
        model.restore(sess, tf.train.latest_checkpoint(checkpoint))
        #print(sess.run('conv1/kernel:0'))
        
        kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1/kernel')
        print(kernel)

        conv1_kernels = sess.run('conv1/kernel:0')
        print(conv1_kernels.shape)
        plot_conv_weights(conv1_kernels)

if __name__ == "__main__":
    latestModel = r'D:\workspace\kaggle-tensor\statoil\tmp\statoil_convnet_model\model.ckpt-500'
    metaFile =  r"D:\workspace\kaggle-tensor\statoil\tmp\statoil_convnet_model\model.ckpt-500.meta"
    checkpoint = r'D:\workspace\kaggle-tensor\statoil\tmp\statoil_convnet_model'
    visualize(metaFile, checkpoint, latestModel)
