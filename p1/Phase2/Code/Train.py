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
from Network.TFSpatialTransformer import *

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
	ImagesBatch = []
	PatchesBatch = []
	GroundTruthBatch = []
	OriginalCornersBatch = []
	
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

		#cv2.imshow("originalPatch", originalPatch)
		#cv2.waitKey(0)

		#cv2.imshow("perturbedPatch", perturbedPatch)
		#cv2.waitKey(0)

		GroundTruth = np.array(perturbations).astype(np.float32)
		GroundTruth = GroundTruth/Perturbation


		#print(np.shape(originalPatch))
		#print(np.shape(perturbedPatch))
		#print(GroundTruth)

		combinedPatch = np.concatenate((originalPatch[:,:,None], perturbedPatch[:,:,None]), axis=2).astype(np.float32)

		combinedPatch /= 255


		I1 = I1.astype(np.float32)
		I1 /= 255

		#print(I1)
		#cv2.imshow("image", I1)
		#cv2.waitKey(0)

		#print(newPoints)




		# Append All Images and Mask
		ImagesBatch.append(I1)
		PatchesBatch.append(combinedPatch)
		GroundTruthBatch.append(GroundTruth)
		OriginalCornersBatch.append(originalPatchInd.reshape((8)).astype(np.float32))

	return ImagesBatch, PatchesBatch, GroundTruthBatch, OriginalCornersBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print("Tensorflow Version " + str(tf.__version__))
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)              

	
def TrainOperation(ImgPH, GroundTruthPH, OriginalCornersPH,ImagesPH, DirNamesTrain,DirNamesVal,NumTrainSamples, PatchSize, Perturbation,
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

	predictedPatch, H4Pt = HomographyModel(ImgPH, OriginalCornersPH, ImagesPH, PatchSize, MiniBatchSize, Perturbation, ModelType)


	with tf.name_scope('Loss'):
		###############################################
		# Fill your loss function of choice here!
		###############################################
		if(ModelType == 'Sup'):

			H4Ptloss = tf.nn.l2_loss(H4Pt - GroundTruthPH)
			valH4PtLossPerEpoch_ph = tf.placeholder(tf.float32,shape=None,name='val_H4Pt_loss_per_epoch')
		else:

			L1loss = tf.math.reduce_sum(tf.abs(ImgPH[:,:,:,1] - predictedPatch[:,:,:,0]))/(PatchSize**2)
			H4Ptloss = tf.nn.l2_loss(H4Pt - GroundTruthPH)

			valL1LossPerEpoch_ph = tf.placeholder(tf.float32,shape=None,name='val_L1_loss_per_epoch')
			valH4PtLossPerEpoch_ph = tf.placeholder(tf.float32,shape=None,name='val_H4Pt_loss_per_epoch')


	with tf.name_scope('Adam'):
		###############################################
		# Fill your optimizer of choice here!
		###############################################
		if(ModelType == 'Sup'):
			Optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(H4Ptloss)
		else:
			Optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(L1loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	if(ModelType == 'Sup'):
		tf.summary.scalar('H4PtLossEveryIter', H4Ptloss/MiniBatchSize)
	else:
		tf.summary.scalar('H4PtLossEveryIter', H4Ptloss/MiniBatchSize)
		tf.summary.scalar('L1LossEveryIter', L1loss/MiniBatchSize)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operation
	MergedSummaryOP = tf.summary.merge_all()

	performance_summaries = []

	if(ModelType == 'Sup'):
		performance_summaries.append(tf.summary.scalar('H4PtValLossPerEpoch', valH4PtLossPerEpoch_ph))
	else:
		performance_summaries.append(tf.summary.scalar('L1ValLossPerEpoch', valL1LossPerEpoch_ph))
		performance_summaries.append(tf.summary.scalar('H4PtValLossPerEpoch', valH4PtLossPerEpoch_ph))

	performance = tf.summary.merge([performance_summaries])


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
				ImagesBatch, PatchBatch, GroundTruthBatch, OriginalCornersBatch = GenerateBatch(BasePath, DirNamesTrain, PatchSize, Perturbation, MiniBatchSize)

				#print(GroundTruthBatch)

				FeedDict = {ImgPH: PatchBatch, GroundTruthPH: GroundTruthBatch, OriginalCornersPH: OriginalCornersBatch, ImagesPH: ImagesBatch}

				if(ModelType == 'Sup'):
					_, H4PtLossThisBatch, Summary, H4Pt_out = sess.run([Optimizer, H4Ptloss, MergedSummaryOP, H4Pt], feed_dict=FeedDict)

				else:
					_, H4PtLossThisBatch, L1LossThisBatch, Summary = sess.run([Optimizer, H4Ptloss,L1loss, MergedSummaryOP], feed_dict=FeedDict)


				#print(testOutput[2])

				#print(testOutput[5])
				#print(testOutput[6])
				# if(PerEpochCounter % 20 == 0):


				# #Full Image
				# cv2.imshow("", testOutput[1][0,:,:])
				# cv2.waitKey(0)

				# #Warped Image
				# cv2.imshow("", testOutput[0][0])
				# cv2.waitKey(0)


				# print(np.shape(testOutput[4]))

				# #Correct Image
				# cv2.imshow("", testOutput[4][0])
				# cv2.waitKey(0)


				#Warped Patch
				#cv2.imshow("",testOutput[3][0])
				#cv2.waitKey(0)

				# Ground Truth
				#cv2.imshow("", np.array(PatchBatch)[0,:,:,1])
				#cv2.waitKey(0)

				
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


			H4PtValLossSum = 0
			L1ValLossSum = 0

			NumIterationsPerEpochVal = int(1000/MiniBatchSize)


			for PerEpochCounter in tqdm(range(NumIterationsPerEpochVal)):
				ImagesBatch, PatchBatch, GroundTruthBatch, OriginalCornersBatch = GenerateBatch(BasePath, DirNamesVal, PatchSize, Perturbation, MiniBatchSize)

				FeedDict = {ImgPH: PatchBatch, GroundTruthPH: GroundTruthBatch, OriginalCornersPH: OriginalCornersBatch, ImagesPH: ImagesBatch}

				if(ModelType == 'Sup'):

					H4PtLossThisBatch = sess.run([H4Ptloss], feed_dict=FeedDict)

					H4PtValLossSum += H4PtLossThisBatch[0]/MiniBatchSize

					print("Epoch " + str(PerEpochCounter) + " H4Pt Loss: " + str(H4PtLossThisBatch[0]/MiniBatchSize))

				else:
					H4PtLossThisBatch, L1LossThisBatch = sess.run([H4Ptloss,L1loss], feed_dict=FeedDict)
					H4PtValLossSum += H4PtLossThisBatch/MiniBatchSize
					L1ValLossSum += L1LossThisBatch/MiniBatchSize








				

	  
			if(ModelType == 'Sup'):

				Summary = sess.run(performance, feed_dict={valH4PtLossPerEpoch_ph:H4PtValLossSum/NumIterationsPerEpochVal})
				Writer.add_summary(Summary,Epochs)
				Writer.flush()

				print("Epoch " + str(Epochs) + " H4Pt Loss: " + str(H4PtValLossSum/NumIterationsPerEpochVal))

			else:
				Summary = sess.run(performance, feed_dict={valH4PtLossPerEpoch_ph:H4PtValLossSum/NumIterationsPerEpochVal, valL1LossPerEpoch_ph:L1ValLossSum/NumIterationsPerEpochVal})
				Writer.add_summary(Summary,Epochs)
				Writer.flush()

				print("Epoch " + str(Epochs) + " H4Pt Val Loss: " + str(H4PtValLossSum/NumIterationsPerEpochVal))
				print("Epoch " + str(Epochs) + " L1 Val Loss: " + str(L1ValLossSum/NumIterationsPerEpochVal))
			

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
	DirNamesTrain, DirNamesVal, SaveCheckPoint, NumTrainSamples = SetupAll(BasePath, CheckPointPath)



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
	OriginalCornersPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8))
	ImagesPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 240,320))
	#WarpedCornersPH = tf.placeholder(tf.int32, shape=(MiniBatchSize,8))
	#CorrectMatrixPH = tf.placeholder(tf.float32, shape=(MiniBatchSize,3,3))
	
	TrainOperation(ImgPH, GroundTruthPH, OriginalCornersPH, ImagesPH, DirNamesTrain, DirNamesVal, NumTrainSamples, PatchSize, Perturbation,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
		
	
if __name__ == '__main__':
	main()
 
