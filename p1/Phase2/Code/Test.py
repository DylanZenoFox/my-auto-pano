#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.DataUtils import *
from Network.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
	"""
	Inputs: 
	BasePath - Path to images
	Outputs:
	ImageSize - Size of the Image
	DataPath - Paths of all images where testing will be run on
	"""   
	# Image Input Shape
	DataPath = []
	NumImages = len(glob.glob(BasePath+'*.jpg'))
	SkipFactor = 1
	for count in range(1,NumImages+1,SkipFactor):
		DataPath.append(BasePath + str(count) + '.jpg')

	return DataPath


def ReadImages(DataPath,PatchSize, Perturbation):
	"""
	Inputs: 
	ImageSize - Size of the Image
	DataPath - Paths of all images where testing will be run on
	Outputs:
	I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
	I1 - Original I1 image for visualization purposes only
	"""
	
	ImageName = DataPath

	I1 = cv2.imread(DataPath, cv2.IMREAD_GRAYSCALE)

	
	if(I1 is None):
		# OpenCV returns empty list if image is not read! 
		print('ERROR: Image I1 cannot be read')
		sys.exit()


	I1 = cv2.resize(I1, (320,240))



	Py = random.randint(Perturbation, np.shape(I1)[0] - (PatchSize + Perturbation))
	Px = random.randint(Perturbation, np.shape(I1)[1] - (PatchSize + Perturbation))

	perturbations = random.sample(xrange(-Perturbation, Perturbation),8)


	#Py = random.randint(50, np.shape(I1)[0] - (PatchSize + 50))
	#Px = random.randint(50, np.shape(I1)[1] - (PatchSize + 50))

	#perturbations = random.sample(xrange(-20, 20),8)

	originalPatchInd = np.array([Px, Py, Px+PatchSize, Py, Px, Py + PatchSize, Px + PatchSize, Py + PatchSize]).reshape((4,2))

	perturbedPatchInd = originalPatchInd + np.array(perturbations).reshape((4,2))

	#cv2.circle(I1, tuple(originalPatchInd[0]), 2, 255, -1)
	#cv2.circle(I1, tuple(originalPatchInd[1]), 2, 255, -1)
	#cv2.circle(I1, tuple(originalPatchInd[2]), 2, 255, -1)
	#cv2.circle(I1, tuple(originalPatchInd[3]), 2, 255, -1)

	#cv2.circle(I1, tuple(perturbedPatchInd[0]), 2, 0, -1)
	#cv2.circle(I1, tuple(perturbedPatchInd[1]), 2, 0, -1)
	#cv2.circle(I1, tuple(perturbedPatchInd[2]), 2, 0, -1)
	#cv2.circle(I1, tuple(perturbedPatchInd[3]), 2, 0, -1)


	matrix = cv2.getPerspectiveTransform(originalPatchInd.astype(np.float32), perturbedPatchInd.astype(np.float32))

	#print(matrix)
	#print(np.linalg.inv(matrix))

	warpedImage = cv2.warpPerspective(I1, np.linalg.inv(matrix), (1000,1000))

	newPoints = cv2.perspectiveTransform(perturbedPatchInd[None,:,:].astype(np.float32), np.linalg.inv(matrix).astype(np.float32))[0,:,:].astype(int)


	originalPatch = I1[Py:Py+PatchSize, Px: Px+PatchSize]

	perturbedPatch = warpedImage[newPoints[0,1]:newPoints[0,1] + PatchSize,newPoints[0,0]:newPoints[0,0] + PatchSize]


	#cv2.imshow("image", I1)
	#cv2.waitKey(0)

	#print(I1)


	#cv2.imshow("warped image", warpedImage)
	#cv2.waitKey(0)

	cv2.imshow("originalPatch", originalPatch)
	cv2.imwrite("./1.jpg", originalPatch)
	cv2.waitKey(0)

	cv2.imshow("perturbedPatch", perturbedPatch)
	cv2.imwrite("./2.jpg", perturbedPatch)

	cv2.waitKey(0)

	GroundTruth = np.array(perturbations).astype(np.float32)
	GroundTruth = GroundTruth/Perturbation


	#print(np.shape(originalPatch))
	#print(np.shape(perturbedPatch))
	#print(GroundTruth)

	combinedPatch = np.concatenate((originalPatch[:,:,None], perturbedPatch[:,:,None]), axis=2).astype(np.float32)

	combinedPatch /= 255


	I1 = I1.astype(np.float32)
	I1 /= 255

	originalPatchInd = originalPatchInd.reshape((8)).astype(np.float32)

	#print(np.shape(I1))
	#print(np.shape(combinedPatch))
	#print(np.shape(GroundTruth))
	#print(np.shape(originalPatchInd))


	return np.expand_dims(I1,axis = 0), np.expand_dims(combinedPatch,axis = 0), np.expand_dims(GroundTruth,axis = 0), np.expand_dims(originalPatchInd,axis = 0)


def TestOperation(PatchPH, OriginalCornersPH,ImagePH, PatchSize,Perturbation,ModelType, ModelPath, DataPath):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	ImageSize is the size of the image
	ModelPath - Path to load trained model from
	DataPath - Paths of all images where testing will be run on
	LabelsPathPred - Path to save predictions
	Outputs:
	Predictions written to ./TxtFiles/PredOut.txt
	"""
	# Predict output with forward pass, MiniBatchSize for Test is 1

	predictedPatch, H4Pt = HomographyModel(PatchPH, OriginalCornersPH , ImagePH , PatchSize, 1, Perturbation, ModelType)

	# Setup Saver
	Saver = tf.train.Saver()

	
	with tf.Session() as sess:
		Saver.restore(sess, ModelPath)
		print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
		
		#OutSaveT = open(LabelsPathPred, 'w')

		EPE = 0
		averageL2 = 0

		for count in tqdm(range(np.size(DataPath))):            
			DataPathNow = DataPath[count]
			Image, Patch, GroundTruth, OriginalCorners = ReadImages(DataPathNow,PatchSize, Perturbation)
			FeedDict = {PatchPH: Patch, OriginalCornersPH: OriginalCorners, ImagePH: Image}
			H4Pt_out = sess.run(H4Pt, FeedDict)

			#print((H4Pt_out * Perturbation))
			#print((GroundTruth * Perturbation))

			ground = (OriginalCorners + (H4Pt_out * Perturbation)).reshape((4,2)).astype(np.int32)
			ground[[2,3]] = ground[[3,2]]
			pred = (OriginalCorners + (GroundTruth * Perturbation)).reshape((4,2)).astype(np.int32)
			pred[[2,3]] = pred[[3,2]]

			#print(ground)
			#print(pred)

			#print(np.square((H4Pt_out) - (GroundTruth))/2)

			averageL2 += np.sum(np.square((H4Pt_out) - (GroundTruth)))/2


			Image = (Image[0]*255).astype('uint8')#cv2.imread(DataPathNow)
			#Image[0] = (Image[0]*255).astype('uint8')

			cv2.polylines(Image,[ground], True, (255,255,255),thickness = 3)
			cv2.polylines(Image,[pred], True, (0,0,0),thickness = 3)
			#cv2.imshow("", Image[0])
			#cv2.waitKey()



			#cv2.imwrite('./Images/ValImages/Unsupervised/' + str(count) + '.jpg', Image)



			EPE += np.sum(np.square((H4Pt_out * Perturbation) - (GroundTruth * Perturbation)))/2




		EPE /= np.size(DataPath)

		averageL2 /= np.size(DataPath)

		print("Average EPE: " + str(EPE))
		print("Average L2: " + str(averageL2))





			#OutSaveT.write(str(PredT)+'\n')
			
		#OutSaveT.close()


		
def main():
	"""
	Inputs: 
	None
	Outputs:
	Prints out the confusion matrix with accuracy
	"""

	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='../Data/Val/', help='Base path of images, Default: ../Data/Val/')
	Parser.add_argument('--ModelPath', default='../Checkpoints/', help='Path to saved Model, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup')
	Parser.add_argument('--PatchSize',type=int, default=128, help='Size for patch extraction, Default=100')
	Parser.add_argument('--Perturbation',type=int, default=32, help='Amount of perturbation of corners when generating data, Default=32')


	Args = Parser.parse_args()
	BasePath = Args.BasePath
	ModelPath = Args.ModelPath
	ModelType = Args.ModelType
	PatchSize = Args.PatchSize
	Perturbation = Args.Perturbation

	# Setup all needed parameters including file reading
	DataPath = SetupAll(BasePath)

	# Define PlaceHolder variables for Input and Predicted output
	PatchPH = tf.placeholder('float', shape=(1, PatchSize, PatchSize, 2))
	#GroundTruthPH = tf.placeholder(tf.float32, shape=(1, 8))
	OriginalCornersPH = tf.placeholder(tf.float32, shape=(1,8))
	ImagePH = tf.placeholder(tf.float32, shape=(1, 240,320))

	TestOperation(PatchPH,OriginalCornersPH,ImagePH, PatchSize, Perturbation, ModelType, ModelPath, DataPath)
	 
if __name__ == '__main__':
	main()
 
