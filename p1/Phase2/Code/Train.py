#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

	
def GenerateBatch(BasePath, DirNamesTrain, PatchSize, Perturbation, MiniBatchSize):
	"""
	Inputs: 
	BasePath - Path to COCO folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	PatchSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels 
	"""
	PatchesBatch = []
	GroundTruthBatch = []
	
	ImageNum = 0
	while ImageNum < MiniBatchSize:
		# Generate random image

		RandIdx = random.randint(0, len(DirNamesTrain)-1)
		
		RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'   
		ImageNum += 1



		##########################################################
		# Add any standardization or data augmentation here!
		##########################################################

		I1 = cv2.imread(RandImageName, cv2.IMREAD_GRAYSCALE)

		I1 = cv2.resize(I1, (320,240))

		Py = random.randint(Perturbation, np.shape(I1)[0] - (PatchSize + Perturbation))
		Px = random.randint(Perturbation, np.shape(I1)[1] - (PatchSize + Perturbation))

		perturbations = random.sample(xrange(-Perturbation, Perturbation),8)


		#Py = random.randint(50, np.shape(I1)[0] - (PatchSize + 50))
		#Px = random.randint(50, np.shape(I1)[1] - (PatchSize + 50))

		#perturbations = random.sample(xrange(-20, 20),8)

		originalPatchInd = np.array([Px, Py, Px+PatchSize, Py, Px, Py + PatchSize, Px + PatchSize, Py + PatchSize]).reshape((4,2))

		perturbedPatchInd = originalPatchInd + np.array(perturbations).reshape((4,2))


		matrix = cv2.getPerspectiveTransform(originalPatchInd.astype(np.float32), perturbedPatchInd.astype(np.float32))

		warpedImage = cv2.warpPerspective(I1, np.linalg.inv(matrix), (1000,1000))

		newPoints = cv2.perspectiveTransform(perturbedPatchInd[None,:,:].astype(np.float32), np.linalg.inv(matrix).astype(np.float32))[0,:,:].astype(int)


		originalPatch = I1[Py:Py+PatchSize, Px: Px+PatchSize]

		perturbedPatch = warpedImage[newPoints[0,1]:newPoints[0,1] + PatchSize,newPoints[0,0]:newPoints[0,0] + PatchSize]


		# cv2.imshow("image", I1)
		# cv2.waitKey(0)


		# cv2.imshow("warped image", warpedImage)
		# cv2.waitKey(0)

		# cv2.imshow("originalPatch", originalPatch)
		# cv2.waitKey(0)

		# cv2.imshow("perturbedPatch", perturbedPatch)
		# cv2.waitKey(0)

		GroundTruth = np.array(perturbations).astype(np.float32)
		GroundTruth = GroundTruth/Perturbation

		#print(np.shape(originalPatch))
		#print(np.shape(perturbedPatch))
		#print(GroundTruth)

		combinedPatch = np.concatenate((originalPatch[:,:,None], perturbedPatch[:,:,None]), axis=2).astype(np.float32)

		combinedPatch /= 255

		# Append All Images and Mask
		PatchesBatch.append(combinedPatch)
		GroundTruthBatch.append(GroundTruth)
		
	return PatchesBatch, GroundTruthBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)              

	
def TrainOperation(ImgPH, GroundTruthPH, DirNamesTrain,NumTrainSamples, PatchSize, Perturbation,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	PatchSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""      
	# Predict output with forward pass

	H4Pt = HomographyModel(ImgPH, PatchSize, MiniBatchSize)


	with tf.name_scope('Loss'):
		###############################################
		# Fill your loss function of choice here!
		###############################################
		if(ModelType is not 'Unsup'):
			loss = tf.nn.l2_loss(H4Pt - GroundTruthPH)*2/8

	with tf.name_scope('Adam'):
		###############################################
		# Fill your optimizer of choice here!
		###############################################
		Optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	tf.summary.scalar('LossEveryIter', loss/MiniBatchSize)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operation
	MergedSummaryOP = tf.summary.merge_all()

	# Setup Saver
	Saver = tf.train.Saver()
	
	with tf.Session() as sess:       
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
			
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				PatchBatch, GroundTruthBatch = GenerateBatch(BasePath, DirNamesTrain, PatchSize, Perturbation, MiniBatchSize)

				#print(GroundTruthBatch)

				FeedDict = {ImgPH: PatchBatch, GroundTruthPH: GroundTruthBatch}
				_, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
				
				# Save checkpoint every some SaveCheckPoint's iterations
				if PerEpochCounter % SaveCheckPoint == 0:
					# Save the Model learnt in this epoch
					SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
					Saver.save(sess,  save_path=SaveName)
					print('\n' + SaveName + ' Model Saved...')

				# Tensorboard
				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()

			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			print('\n' + SaveName + ' Model Saved...')
			

def main():
	"""
	Inputs: 
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='../Data', help='Base path of images, Default: ../Data')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
	Parser.add_argument('--PatchSize',type=int, default=128, help='Size for patch extraction, Default=100')
	Parser.add_argument('--Perturbation',type=int, default=32, help='Amount of perturbation of corners when generating data, Default=32')


	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType
	PatchSize = Args.PatchSize
	Perturbation = Args.Perturbation

	# Setup all needed parameters including file reading
	DirNamesTrain, SaveCheckPoint, NumTrainSamples = SetupAll(BasePath, CheckPointPath)



	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.placeholder(tf.float32, shape = (MiniBatchSize, PatchSize, PatchSize, 2))
	GroundTruthPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8))
	
	TrainOperation(ImgPH, GroundTruthPH, DirNamesTrain, NumTrainSamples, PatchSize, Perturbation,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
		
	
if __name__ == '__main__':
	main()
 
