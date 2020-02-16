"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, PatchSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    PatchSize - Size of the Patch
    Outputs:
    H4Pt - Predicted point deviations
    """

    #############################
    # Fill your network here!
    #############################

    conv1 = tf.layers.conv2d(inputs= Img, filters= 64, kernel_size=3, strides = 1, padding="same", data_format = "channels_last", activation="relu")
    batch_norm1 = tf.layers.batch_normalization(inputs = conv1)
    conv2 = tf.layers.conv2d(inputs= batch_norm1, filters= 64, kernel_size=3, strides = 1, padding="same", data_format = "channels_last", activation="relu")
    batch_norm2 = tf.layers.batch_normalization(inputs = conv2)
    max_pool1 = tf.layers.max_pooling2d(inputs = batch_norm2,pool_size=2, strides=2, data_format = "channels_last", padding="same")
    conv3 = tf.layers.conv2d(inputs= max_pool1, filters= 64, kernel_size=3, strides = 1, padding="same", data_format = "channels_last", activation="relu")
    batch_norm3 = tf.layers.batch_normalization(inputs = conv3)
    conv4 = tf.layers.conv2d(inputs= batch_norm3, filters= 64, kernel_size=3, strides = 1, padding="same", data_format = "channels_last", activation="relu")
    batch_norm4 = tf.layers.batch_normalization(inputs = conv4)
    max_pool2 = tf.layers.max_pooling2d(inputs = batch_norm4,pool_size=2, strides=2, data_format = "channels_last", padding = "same")
    conv5 = tf.layers.conv2d(inputs= max_pool2, filters= 128, kernel_size=3, strides = 1, padding="same", data_format = "channels_last", activation="relu")
    batch_norm5 = tf.layers.batch_normalization(inputs = conv5)
    conv6 = tf.layers.conv2d(inputs= batch_norm5, filters= 128, kernel_size=3, strides = 1, padding="same", data_format = "channels_last", activation="relu")
    batch_norm6 = tf.layers.batch_normalization(inputs = conv6)
    dropout1 = tf.layers.dropout(inputs = batch_norm6, rate = 0.5)
    shape = dropout1.get_shape().as_list()
    flatten = tf.reshape(dropout1, [MiniBatchSize, tf.shape(dropout1)[1] * tf.shape(dropout1)[2] * tf.shape(dropout1)[3]])
    dense1 = tf.layers.dense(inputs = flatten, units = 1024, activation="relu")
    dropout2 = tf.layers.dropout(inputs = dense1, rate = 0.5)
    dense2 = tf.layers.dense(inputs= dropout2, units = 8, activation = None)   

    H4Pt = dense2

    print(H4Pt)

    return H4Pt

