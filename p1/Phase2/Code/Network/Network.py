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
import cv2
import sys
import numpy as np
from TensorDST import *
from TFSpatialTransformer import *
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, OriginalCorners , Images ,PatchSize, MiniBatchSize, Perturbation, ModelType = 'Sup'):
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


    if(ModelType == 'Sup'):
        return SupervisedHomographyModel(Img, PatchSize, MiniBatchSize)
    else:
        return UnsupervisedHomographyModel(Img,Images, PatchSize, MiniBatchSize, OriginalCorners, Perturbation)


def SupervisedHomographyModel(Img, PatchSize, MiniBatchSize):



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
    dropout1 = tf.layers.dropout(inputs = batch_norm6, rate = 0.1)
    shape = dropout1.get_shape().as_list()
    flatten = tf.reshape(dropout1, [MiniBatchSize, tf.shape(dropout1)[1] * tf.shape(dropout1)[2] * tf.shape(dropout1)[3]])
    dense1 = tf.layers.dense(inputs = flatten, units = 1024, activation="relu")
    dropout2 = tf.layers.dropout(inputs = dense1, rate = 0.1)
    dense2 = tf.layers.dense(inputs= dropout2, units = 8, activation = None)   

    H4Pt = dense2
    
    return None, H4Pt



def UnsupervisedHomographyModel(Img, FullImage, PatchSize, MiniBatchSize, OriginalCorners, Perturbation):

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
    dropout1 = tf.layers.dropout(inputs = batch_norm6, rate = 0.1)
    shape = dropout1.get_shape().as_list()
    flatten = tf.reshape(dropout1, [MiniBatchSize, tf.shape(dropout1)[1] * tf.shape(dropout1)[2] * tf.shape(dropout1)[3]])
    dense1 = tf.layers.dense(inputs = flatten, units = 1024, activation="relu")
    dropout2 = tf.layers.dropout(inputs = dense1, rate = 0.1)
    dense2 = tf.layers.dense(inputs= dropout2, units = 8, activation = None)   

    H4Pt = dense2

    PredictedCorners = H4Pt * Perturbation + OriginalCorners
    #print(WarpedCorners)

    # Retrieve H4Pt

    homography = TensorDST(H4Pt,OriginalCorners, Perturbation, MiniBatchSize)


    #print(FullImage[:,:,:,None])
    #print(homography[:,None,:])

    homography = tf.reshape(homography,[-1,3,3])

    #invhom = tf.reshape(tf.linalg.inv(homography), [-1,9])

    #print(CorrectMatrix[0])
    #print(homography)


    warpedImage = batch_transformer(FullImage[:,:,:,None], homography[:,None,:], (1000, 1000))

    #correct = batch_transformer(FullImage[:,:,:,None], CorrectMatrix[:,None,:], (1000, 1000))

    #homography = tf.reshape(homography,[-1,3,3])

    #print(warpedImage)


    #print(homography[0,:,:])
    #print(OriginalCorners)
    #print(OriginalCorners[0,:2])
    #print(tf.concat([OriginalCorners[0,:2],np.array([1])], axis = 0))

    patches = []
    for i in range(MiniBatchSize):

        topleft = tf.linalg.matvec(homography[i,:,:], tf.concat([PredictedCorners[i,:2],np.array([1])], axis = 0))

        topleft /= topleft[2]

        topleft = tf.cast(topleft, dtype=tf.int32)

        warpedPatch = warpedImage[0][i,topleft[1]:topleft[1] + PatchSize,topleft[0]:topleft[0] + PatchSize,:]

        print(warpedPatch)

        patches.append(warpedPatch)


    warpedPatch = tf.stack(patches, axis = 0)

    #print(warpedPatch)

    #print(homography)



    # patches = []

    # for i in range(MiniBatchSize):

    #     print(tf.reshape(PredictedCorners,[-1,2,4])[None,i,:,:])

    #     newPoints = cv2.perspectiveTransform(tf.reshape(PredictedCorners,[-1,2,4])[None,i,:,:], tf.linalg.inv(homography))[i,:,:].astype(int)

    #     perturbedPatch = warpedImage[newPoints[0,1]:newPoints[0,1] + PatchSize,newPoints[0,0]:newPoints[0,0] + PatchSize]

    #     patches.append(perturbedPatch)

    # output = tf.stack(patches, axis =ca 0)




    #print(tf.reshape(OriginalCorners,[-1,4,2])[:,2])

    #p = tf.concat([tf.reshape(OriginalCorners,[-1,4,2])[:,0], tf.transpose([tf.ones(MiniBatchSize)])], axis = 1)

    #newpoint = tf.transpose(tf.matmul(tf.reshape(homography,[-1,3,3]), p[:,:,None]),perm = (0,2,1))[:,0,:]

    #focal = newpoint[:,2]

    #points = tf.transpose(tf.transpose(newpoint)/focal)[:,:2]

    #print(points)

    return warpedPatch, H4Pt#, (warpedImage[0], FullImage, homography, warpedPatch, correct[0],PredictedCorners[0,:2], topleft)


