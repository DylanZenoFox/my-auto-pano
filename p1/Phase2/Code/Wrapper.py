#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import tensorflow as tf
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
from Utils import *
# Add any python libraries here



def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageSetBasePath', default="../../Phase1/Data/Train/Set1", help='Number of best features to extract from each image, Default: ../Data/Train/Set1')
	Parser.add_argument('--NumFeatures', default=300,type=int ,help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--ModelPath', default='../Checkpoints/', help='Path to saved Model, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup')

	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	ImageSetBasePath = Args.ImageSetBasePath
	ModelPath = Args.ModelPath
	ModelType = Args.ModelType


	images = read_images(ImageSetBasePath)
	colored_images = read_images(ImageSetBasePath, color=True)

	num_images = len(images)

	corners = detect_corners(images)

	corner_points = get_corner_points(corners)

	best_corners = []

	for i in range(len(colored_images)):

		b = anms(corner_points[i], NumFeatures)
		best_corners.append(b)


	match_matrix = np.zeros((num_images,num_images))
	best_match_dict = {}


	for i in range(num_images):

		#cv2.imshow("", images[i])
		#cv2.waitKey(0)



		for j in range(num_images):
			if(i != j):

				features1 = compute_features(images[i], best_corners[i])
				features2 = compute_features(images[j], best_corners[j])

				matches = feature_match(images[i], features1, images[j],features2)

				homography, num_matches, best_matches = ransac(images[i], images[j], matches)

				match_matrix[i,j] = num_matches
				best_match_dict[(i,j)] = best_matches
				#homography_matrix[i,j] = homography

				#print(num_matches)

	top_match = np.where(match_matrix==match_matrix.max())

	top_match = (top_match[0].tolist()[0], top_match[1].tolist()[0])

	best_matches = best_match_dict[top_match]

	print(best_matches)

	for match in range(len(best_matches[0])):

		if(verify_point(images[top_match[0]], best_matches[0][match].pt) and verify_point(images[top_match[1]], best_matches[1][match].pt)):

			point1 = np.array(best_matches[0][match].pt).astype(np.int32)
			point2 = np.array(best_matches[1][match].pt).astype(np.int32)

			print(point1)
			print(point2)

			print(np.shape(images[top_match[0]]))
			print(np.shape(images[top_match[0]]))

			break




	originalPatch = images[top_match[0]][point1[1] - 64: point1[1] + 64, point1[0] - 64: point1[0] + 64]

	perturbedPatch = images[top_match[1]][point2[1] - 64: point2[1] + 64, point2[0] - 64: point2[0] + 64]

	original_points = np.array([point1[0] - 64, point1[1] - 64, point1[0] + 64, point1[1] - 64, point1[0] - 64,point1[1] + 64, point1[0] + 64,point1[1] + 64]).reshape(4,2)
	print(original_points)


	combinedPatch = np.concatenate((originalPatch[:,:,None], perturbedPatch[:,:,None]), axis=2).astype(np.float32)


	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

	PatchPH = tf.placeholder('float', shape=(1, 128, 128, 2))
	OriginalCornersPH = tf.placeholder(tf.float32, shape=(1,8))
	ImagePH = tf.placeholder(tf.float32, shape=(1, 240,320))

	predictedPatch, H4Pt = HomographyModel(PatchPH, OriginalCornersPH , ImagePH , 128, 1, 32, 'Sup')

	# Setup Saver
	Saver = tf.train.Saver()

	with tf.Session() as sess:
		Saver.restore(sess, ModelPath)

		FeedDict = {PatchPH: np.expand_dims(combinedPatch/255,axis = 0)}#, OriginalCornersPH: OriginalCorners, ImagePH: Image}
		H4Pt_out = sess.run(H4Pt, FeedDict)

		pred_perturations = np.reshape(H4Pt_out[0]* 32,(4,2)).astype(np.int32)
		print(pred_perturations)


	predicted = pred_perturations + original_points
	print(predicted)

	homography = cv2.getPerspectiveTransform(predicted.astype(np.float32), original_points.astype(np.float32))


	print(homography)

	result = warpTwoImages(images[top_match[1]], images[top_match[0]], homography)

	cv2.imshow("", result)
	cv2.waitKey(0)


	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


def warpTwoImages(img2, img1, H):
	'''warp img1 to img2 with homograph H'''
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

	result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
	try:
		for y in range(t[1],h1+t[1]):
			for x in range(t[0],w1+t[0]):
				if result[y][x][0] == 0 and result[y][x][1] == 0 and result[y][x][2] == 0:
					for c in range(3):
						result[y][x][c] = img1[y-t[1]][x-t[0]][c]
	except IndexError:
		for y in range(t[1],h1+t[1]):
			for x in range(t[0],w1+t[0]):
				if result[y][x] == 0:
					result[y][x] = img1[y-t[1]][x-t[0]]

#    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
	return result




def verify_point(image, point_tup, size = 64):

	shape = np.shape(image)

	#print(point_tup)
	#print(shape)
	#print(point_tup[0])
	#print(point_tup[1])

	if(point_tup[0] > size and point_tup[0] < shape[0] - size and point_tup[1] > size and point_tup[1] < shape[1] - size):
		return True
	else:
		return False













def read_images(path, color=False):
	images = []
	for file in os.listdir(path):
		if(file.endswith(".png") or file.endswith(".jpg")):
			file = path + "/" + file
			if color:
				image = cv2.imread(file,cv2.IMREAD_COLOR)
			else:
				image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
			images.append(image)
	return images

def display_results(images,name='image'):
	for image in images:
		cv2.imshow(name, image)
		cv2.waitKey(0)


	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	corners = []

def detect_corners(input):
	if isinstance(input, list):
		return [detect_corners(image) for image in input]
	img32 = np.float32(input)
	dst = cv2.cornerHarris(img32, 2,3,0.04)
	dst[dst<1000000] = 0
	return dst

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

def get_corner_points(input):
	if isinstance(input, list):
		return [get_corner_points(image) for image in input]
	points = []
	y_max = input.shape[0]
	x_max = input.shape[1]
	for i in range(20, y_max-20):
		for j in range(20, x_max-20):
			if input[i][j] > 0.0:
				points.append((i,j,input[i][j]))
	return points


def show_corner_points(image, points):
	for point in points:
		cv2.circle(image, (int(point[1]), int(point[0])), 1, (0,0,0))

	cv2.imshow("Cornerpoints",image)
	cv2.waitKey(0)

def anms(points, Nbest):

	r = [10e+20] * len(points)

	for i in range(len(points)):

		for j in range(len(points)):

			if(points[j][2] > points[i][2]):

				ED = (points[j][0] - points[i][0])**2 + (points[j][1] - points[i][1])**2

				if(ED < r[i]):
					r[i] = ED

	r_points = [(r[i], points[i]) for i in list(range(len(points)))]

	r_points_sorted = sorted(r_points, reverse = True)


	#print(r_points_sorted)

	return [tup[1] for tup in r_points_sorted[:Nbest]]


def compute_features(image, points, size = 40):

	size = size/2

	im_shape = np.shape(image)

	#print(im_shape)
	#print("hi")
	#print(size)

	features = []

	for point in points:

		if(point[0] - size < 0 or point[0] + size > im_shape[0]
			or point[1] - size < 0 or point[1] + size > im_shape[1]):

			continue

		patch = image[point[0]-size:point[0]+size, point[1]-size:point[1] + size]

		patch = cv2.GaussianBlur(patch, (5,5), sigmaX = 2)

		patch = cv2.resize(patch, (8,8))

		patch = np.reshape(patch, (64))

		patch = (patch - np.mean(patch))/ np.std(patch)

		features.append((point[1], point[0], patch))

	return features



def feature_match(image1, features1, image2, features2, threshold = 0.7):

	#print(len(features1))
	#print(len(features2))

	keypoints1 = []
	keypoints2 = []
	matches = []
	idx = 0

	for f1 in features1:

		SSD = []

		for f2 in features2:

			SSD.append( ((np.sum((f1[2] - f2[2])**2)), (f1[0], f1[1]), (f2[0], f2[1])))

		sorted_SSD = sorted(SSD)

		#print(sorted_SSD)

		best_match = sorted_SSD[0]
		second_best_match = sorted_SSD[1]

		if(best_match[0]/second_best_match[0] < threshold):

			#print(best_match[1][0])

			keypoints1.append(cv2.KeyPoint(float(best_match[1][0]), float(best_match[1][1]), 10))
			keypoints2.append(cv2.KeyPoint(float(best_match[2][0]), float(best_match[2][1]), 10))
			matches.append(cv2.DMatch(idx,idx, best_match[0]))
			idx += 1

	#print(len(keypoints1))
	#print(len(keypoints2))

	#print(np.shape(image1))
	#print(np.shape(image2))

	#drawMatches(image1, keypoints1, image2, keypoints2, matches)

	return (keypoints1, keypoints2, matches)




def ransac(image1, image2, matches, Nmax = 500, threshold = 5, inlier_target = 0.8):

	num_matches = len(matches[0])

	if(num_matches < 4):
		print("Images do not overlap")
		return None, 0 , None

	largest_inliers = []
	best_homography = []

	for i in range(Nmax):

		points = random.sample(range(0, num_matches), 4)

		p1 = [list(keypoint.pt) for keypoint in matches[0]]
		p2 = [list(keypoint.pt) for keypoint in matches[1]]

		sampledPoints1 = np.array([p1[i] for i in points]).astype(np.float32)
		sampledPoints2 = np.array([p2[i] for i in points]).astype(np.float32)

		matrix = cv2.getPerspectiveTransform(sampledPoints2, sampledPoints1)

		newpoints = cv2.perspectiveTransform(np.array([p2]).astype(np.float32),  matrix)

		inliers = []

		for i, j in enumerate(newpoints.tolist()[0]):

			SSD = (p1[i][0] - j[0])**2 + (p1[i][1] - j[1])**2

			if(SSD < threshold):
				inliers.append((SSD,i))

		if(len(inliers) > len(largest_inliers)):
			#print("SUCCESS")
			#print(len(inliers))
			largest_inliers = inliers

			print(largest_inliers)

			best_homography = matrix

			if(len(largest_inliers) > inlier_target * num_matches):
				#print("FINISHING")


				break

	inliers_matches = [inlier[1] for inlier in largest_inliers]

	#drawMatches(image1, [matches[0][i] for i in inliers_matches], image2, [matches[1][i] for i in inliers_matches], [cv2.DMatch(i,i,matches[2][j].distance) for i,j in enumerate(inliers_matches)])

	best_inliers = sorted(largest_inliers)
	

	num_matches = len(largest_inliers)

	if(num_matches < 5):
		print("NOT A GOOD HOMOGRAPHY")
		return None, 0, None



	return best_homography, num_matches, ([matches[0][i] for i in inliers_matches],[matches[1][i] for i in inliers_matches],[cv2.DMatch(i,i,matches[2][j].distance) for i,j in enumerate(inliers_matches)])

    
if __name__ == '__main__':
    main()
 
