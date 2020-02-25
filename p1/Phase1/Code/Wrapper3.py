#!/usr/bin/evn python

#Arcticfox on stackexchange

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
import argparse
import os
import random
import matplotlib.pyplot as plt
from Utils import *
# Add any python libraries here

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


def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageSetBasePath', default="../Data/Test/Phase1/TestSet1", help='Number of best features to extract from each image, Default: ../Data/Train/Set1')
	Parser.add_argument('--NumFeatures', default=300,type=int ,help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--System', default="mac", help="Sets system for visualization, Options: 'linux', 'mac'")

	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	ImageSetBasePath = Args.ImageSetBasePath
	System = Args.System



	# images = read_images(ImageSetBasePath)
	# colored_images = read_images(ImageSetBasePath, color=True)

	# num_images = len(images)

	# core = None

	# while(num_images > 1):

	# 	corners = detect_corners(images)

	# 	corner_points = get_corner_points(corners)

	# 	best_corners = []

	# 	for i in range(len(colored_images)):

	# 		b = anms(corner_points[i], NumFeatures)
	# 		best_corners.append(b)

	# 	match_matrix = np.zeros((num_images,num_images))
	# 	homography_matrix = np.zeros((num_images, num_images, 3, 3))

	# 	for i in range(num_images):
	# 		for j in range(num_images):
	# 			if(i != j):

	# 				features1 = compute_features(images[i], best_corners[i])
	# 				features2 = compute_features(images[j], best_corners[j])

	# 				matches = feature_match(images[i], features1, images[j],features2)

	# 				homography, num_matches, best_matches = ransac(images[i], images[j], matches)

	#  				match_matrix[i,j] = num_matches
	#  				homography_matrix[i,j] = homography

	#  				#print(i)
	#  				#print(j)



	#  	print(match_matrix)

	#  	print(np.sum(match_matrix,axis = 0) + np.sum(match_matrix, axis = 1))


	#  	if(core == None):
	#  		coreImage = np.argmax(np.sum(match_matrix,axis = 0) + np.sum(match_matrix, axis = 1))
	#  	else:
	#  		coreImage = core


	#  	imageToFuse = np.argmax(match_matrix[coreImage])

	#  	print(coreImage)
	#  	print(imageToFuse)

	#  	homography = homography_matrix[coreImage, imageToFuse]
	#  	#homography = homography_matrix[imageToFuse, coreImage]


	#  	#resultImageColored = warpTwoImages(colored_images[coreImage], colored_images[imageToFuse], homography)
	#  	#resultImage = warpTwoImages(images[coreImage], images[imageToFuse], homography)

	#  	resultImageColored = warpTwoImages(colored_images[imageToFuse], colored_images[coreImage], homography)
	#  	resultImage = warpTwoImages(images[imageToFuse], images[coreImage], homography)

	#  	cv2.imwrite("./merged" + str(num_images) + ".jpg", resultImageColored)

	#  	if(imageToFuse > coreImage):

	#  		images.pop(imageToFuse)
	#  		images.pop(coreImage)
	#  		colored_images.pop(imageToFuse)
	#  		colored_images.pop(coreImage)
	#  	else:
	#  		images.pop(coreImage)
	#  		images.pop(imageToFuse)
	#  		colored_images.pop(coreImage)
	#  		colored_images.pop(imageToFuse)

	#  	images.append(resultImage)
	#  	colored_images.append(resultImageColored)

	#  	num_images = len(images)

	#  	core = num_images - 1


	images = read_images(ImageSetBasePath)
	colored_images = read_images(ImageSetBasePath, color=True)

	num_images = len(images)

	original_lenth = len(images)
	while(num_images > 1):

		corners = detect_corners(images)

		corner_points = get_corner_points(corners)

		best_corners = []

		for i in range(len(colored_images)):

			b = anms(corner_points[i], NumFeatures)
			best_corners.append(b)

		match_matrix = np.zeros((num_images,num_images))
		homography_matrix = np.zeros((num_images, num_images, 3, 3))

		for i in range(num_images):
			for j in range(num_images):
				if(i != j):

					features1 = compute_features(images[i], best_corners[i])
					features2 = compute_features(images[j], best_corners[j])

					matches = feature_match(images[i], features1, images[j],features2)

					homography, num_matches, best_matches = ransac(images[i], images[j], matches)

	 				match_matrix[i,j] = num_matches
	 				homography_matrix[i,j] = homography


	 	print(match_matrix)

	 	print(np.sum(match_matrix,axis = 0) + np.sum(match_matrix, axis = 1))

		if np.max(np.sum(match_matrix,axis = 0) + np.sum(match_matrix, axis = 1)) < 100:
			break

#		if num_images == original_lenth:
#			coreImage = original_lenth/2
		coreImage = np.argmax(np.sum(match_matrix,axis = 0) + np.sum(match_matrix, axis = 1))
#		else:
#			coreImage = -1


	 	imageToFuse = np.argmax(match_matrix[coreImage])

	 	print(coreImage)
	 	print(imageToFuse)

	 	homography = homography_matrix[coreImage, imageToFuse]
	 	#homography = homography_matrix[imageToFuse, coreImage]


	 	#resultImageColored = warpTwoImages(colored_images[coreImage], colored_images[imageToFuse], homography)
	 	#resultImage = warpTwoImages(images[coreImage], images[imageToFuse], homography)

	 	resultImageColored = warpTwoImages(colored_images[imageToFuse], colored_images[coreImage], homography)
	 	resultImage = warpTwoImages(images[imageToFuse], images[coreImage], homography)

	 	cv2.imwrite("./merged" + str(num_images) + ".jpg", resultImageColored)

	 	if(imageToFuse > coreImage):

	 		images.pop(imageToFuse)
	 		images.pop(coreImage)
	 		colored_images.pop(imageToFuse)
	 		colored_images.pop(coreImage)
	 	else:
	 		images.pop(coreImage)
	 		images.pop(imageToFuse)
	 		colored_images.pop(coreImage)
	 		colored_images.pop(imageToFuse)

	 	images.append(resultImage)
	 	colored_images.append(resultImageColored)

	 	num_images = len(images)
















	"""
	Read a set of images for Panorama stitching
	"""

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



def feature_match(image1, features1, image2, features2, threshold = 0.5):

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
				inliers.append(i)

		if(len(inliers) > len(largest_inliers)):
			#print("SUCCESS")
			#print(len(inliers))
			largest_inliers = inliers

			best_homography = matrix

			if(len(largest_inliers) > inlier_target * num_matches):
				#print("FINISHING")


				break

	# drawMatches(image1, [matches[0][i] for i in largest_inliers], image2, [matches[1][i] for i in largest_inliers], [cv2.DMatch(i,i,matches[2][j].distance) for i,j in enumerate(largest_inliers)])

	# homographies = []

	# for i in range(Nmax):

	# 	points = random.sample(largest_inliers, 4)

	# 	p1 = [list(keypoint.pt) for keypoint in matches[0]]
	# 	p2 = [list(keypoint.pt) for keypoint in matches[1]]

	# 	sampledPoints1 = np.array([p1[i] for i in points]).astype(np.float32)
	# 	sampledPoints2 = np.array([p2[i] for i in points]).astype(np.float32)

	# 	matrix = cv2.getPerspectiveTransform(sampledPoints2, sampledPoints1)

	# 	homographies.append(matrix)

	# 	print(matrix)


	# print("BEST AND AVERAGE")
	# avg_homography = np.mean(np.stack(homographies),axis = 0)
	# print(avg_homography)
	# print(best_homography)
	if best_homography == []:
		return None, 0, None

	return best_homography, num_matches, ([matches[0][i] for i in largest_inliers],[matches[1][i] for i in largest_inliers],[cv2.DMatch(i,i,matches[2][j].distance) for i,j in enumerate(largest_inliers)])



def stitchImages(image1, image2, homography):


	cv2.imshow("", image2)
	cv2.waitKey()

	cv2.imshow("",cv2.warpPerspective(image1, homography, (1000,1000)))
	cv2.waitKey()




















































































####################################################################################3
def anms_algorithm(input):
	ris = []
	for i in range(len(input)):
		ris.append([])
		n_strong = len(input[i])
		for j in range(n_strong):
			ri = 10e+20
			for k in range(n_strong):
				distance = 10e+20
				if input[i][k][2] > input[i][j][2]:
#					print(amns[i][k])
#					print(amns[i][j])
					distance = ((amns[i][k][0]-amns[i][j][0])**2 + (amns[i][k][1]-amns[i][j][1])**2)
#					print(distance)
				if distance < ri:
					ri = distance
#					print(ri)
#			print(ri)
			ris[i].append((amns[i][j][0],amns[i][j][1],ri))
#		print(ris[i])


def run_anms(input, n_best):

	corner_points = get_corner_points(input)
	amns = []
	n_best = NumFeatures
	ris = []
	best_corners = []

# 	for i in range(len(corners)):
# 		x_max = corners[i].shape[0]
# 		y_max = corners[i].shape[1]
# 		amns.append([])
# 		for j in range(20,x_max-20):
# 			for k in range(20,y_max-20):
# 				if corners[i][j][k] > 0.0:
# #					print((j,k,corners[i][j][k]))
# 					amns[i].append((j,k,corners[i][j][k]))



	for i in range(len(ris)):
		ris[i] = sorted(ris[i], key = lambda x: x[2], reverse=True)
		ris[i] = ris[i][:n_best]

	for i in range(len(corners)):
		best_corner = np.zeros(corners[i].shape)
		for ri in ris[i]:
			best_corner[ri[0]][ri[1]] = ri[2]
#			print(best_corner[ri[0]][ri[1]])
		best_corners.append(best_corner)

	for best_corner in best_corners:
		cv2.imshow("best_corner", best_corner)
		cv2.waitKey(0)


	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	descriptors = []

	for i in range(len(images)):
		descriptors.append([])
		image = images[i]
		best_corners = ris[i]
		for j in range(len(best_corners)):
			good_corner = best_corners[j]
			y = good_corner[0]
			x = good_corner[1]
			desc_matrix = image[y-20:y+20,x-20:x+20]

			blur = cv2.GaussianBlur(desc_matrix, (5,5), 1)

			sub = cv2.resize(blur, (8,8))

			flat = sub.flatten()

			mean = np.mean(flat)
			mean_0 = flat - mean

			std = np.std(mean_0)
			std_1 = mean_0/std

			descriptors[i].append(((x,y),std_1))

			#Checking feature correctness

			#print(std_1.size)
			#plt.imshow(np.reshape(std_1,(8,8)))
			#plt.show()




	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""

	matches = {}
	threshold = .5

	for i in range(len(descriptors)):
		for j in range(i+1,len(descriptors)):
			# each pair of images will have 3 lists where the first list is a list
			# of points in image 1, the second is a list of corresponding matching
			# points in image 2 and the third is a list of corresponding distances of that match
			matches.update({(i,j):[[],[],[]]})
			for m in range(len(descriptors[i])):
				best_distance = 5000
				best_point = None
				second_best_distance = 10000
				second_best_point = None
				# for each corner m in image i, look at each corner n in image j
				# if the distance of those descriptors is less than m's distance
				# to any other n, save that distance and n
				for n in range(len(descriptors[j])):
					if (descriptors[i][m][0] not in matches[(i,j)][0] and \
					descriptors[j][n][0] not in matches[(i,j)][1]):
						difference = descriptors[i][m][1] - descriptors[j][n][1]
						square = np.square(difference)
						distance = np.sum(square)
						if distance < best_distance:
							second_best_distance = best_distance
							second_best_point = best_point
							best_distance = distance
							best_point = n

				# for each corner m in image i, if the ratio of its distance
				# to the best corner n of image j is less than a threshold,
				# save those points as a match
				if best_distance/second_best_distance < threshold:
					matches[(i,j)][0].append(descriptors[i][m][0])
					matches[(i,j)][1].append(descriptors[j][best_point][0])
					matches[(i,j)][2].append(best_distance)


	for i in range(len(descriptors)):
		for j in range(i+1, len(descriptors)):
			match_lists = matches[(i,j)]
	#		match_lists[0] = [(x,y) for (y,x) in match_lists[0]]
	#		match_lists[1] = [(x,y) for (y,x) in match_lists[1]]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[1]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert(match_lists[0])
				kp2 = cv2.KeyPoint_convert(match_lists[1])

			distances = match_lists[2]
			Dmatches = []
			for k in range(len(kp1)):
				# convert the distances to DMatches for use with drawMatches
				Dmatches.append(cv2.DMatch(k,k,distances[k]))

			if System == "linux":
				drawn_matches = drawMatches(images[j],kp2,images[i],kp1,Dmatches)

			if System == "mac":
				drawn_matches = cv2.drawMatches(images[i],kp1,images[j],kp2,Dmatches,None)

			cv2.imshow("matches", drawn_matches)
			cv2.waitKey(0)

	"""
	Refine: RANSAC, Estimate Homography
	"""

	n_max = 1000
	tau = 100
	threshold = .95

	filtered_matches = {}

	for p in range(len(descriptors)):
		for q in range(p+1,len(descriptors)):
			image_pair = (p,q)
			filtered_matches.update({image_pair:[[],[],[],[],[],[]]})

			best_h = np.zeros((3,3))
			best_in_num = 0
			best_inliers_1 = []
			best_inliers_2 = []
			best_distances = []
			best_inliers_prime = []
			best_score = 0

			num_matches = len(matches[image_pair][0])
			im1_points = matches[image_pair][0]
			im2_points = matches[image_pair][1]
			im_distances = matches[image_pair][2]
			for n in range(n_max):
				if num_matches < 4:
					continue
				test_points = random.sample(range(num_matches),4)
				im1_test = [im1_points[num] for num in test_points]
				im2_test = [im2_points[num] for num in test_points]

				im1_array = np.zeros((4,2),dtype=np.float32)
				im2_array = np.zeros((4,2),dtype=np.float32)

				for point in range(4):
					#im1_array[point] = im1_test[point][::-1]
					#im2_array[point] = im2_test[point][::-1]
					for coord in range(2):
						im1_array[point][coord] = im1_test[point][coord]
						im2_array[point][coord] = im2_test[point][coord]

				h = cv2.getPerspectiveTransform(im1_array, im2_array)

				in_num = 0
				inliers_1 = []
				inliers_2 = []
				distances = []
				inliers_prime = []
				for i in range(len(im1_points)):
					pi = np.array([[im1_points[i][0]],[im1_points[i][1]],[1]],dtype=np.float32)
					pi_prime = np.array([[im2_points[i][0]],[im2_points[i][1]],[1]],dtype=np.float32)
					hpi = np.dot(h,pi)
					difference = np.subtract(hpi,pi_prime)
					square = np.square(difference)
					sum = np.sum(square)
					if sum < tau:
						in_num += 1
						inliers_1.append(im1_points[i])
						inliers_2.append(im2_points[i])
						distances.append(im_distances[i])
						inliers_prime.append((hpi.item(0),hpi.item(1)))

				if in_num > best_in_num:
					best_h = h
					best_in_num = in_num
					best_inliers_1 = inliers_1
					best_inliers_2 = inliers_2
					best_distances = distances
					best_inliers_prime = inliers_prime
					best_score = float(best_in_num)/float(len(im1_points))
					if best_score > threshold:
						break

			new_h = np.zeros((3,3))
			for i in range(100):
				if best_in_num < 4:
					break
				test_points = random.sample(range(best_in_num),4)
				im1_test = [best_inliers_1[num] for num in test_points]
				im2_test = [best_inliers_2[num] for num in test_points]

				im1_array = np.zeros((4,2),dtype=np.float32)
				im2_array = np.zeros((4,2),dtype=np.float32)

				for point in range(4):
					for coord in range(2):
						im1_array[point][coord] = im1_test[point][coord]
						im2_array[point][coord] = im2_test[point][coord]

				h = cv2.getPerspectiveTransform(im1_array, im2_array)

				new_h = np.add(new_h,h)

			new_h = new_h * 0.01

			best_inliers_prime = []
			for i in range(len(best_inliers_1)):
				pi = np.array([[best_inliers_1[i][0]],[best_inliers_1[i][1]],[1]],dtype=np.float32)
				pi_prime = np.array([[best_inliers_2[i][0]],[best_inliers_2[i][1]],[1]],dtype=np.float32)
				hpi = np.dot(best_h,pi)
				best_inliers_prime.append((hpi.item(0),hpi.item(1)))

			filtered_matches[image_pair][0] = best_inliers_1
			filtered_matches[image_pair][1] = best_inliers_2
			filtered_matches[image_pair][2] = best_distances
			filtered_matches[image_pair][3] = best_inliers_prime
			filtered_matches[image_pair][4] = best_h
			filtered_matches[image_pair][5] = best_score

	for i in range(len(descriptors)):
		for j in range(i+1, len(descriptors)):

			match_lists = filtered_matches[(i,j)]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[1]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert(match_lists[0])
				kp2 = cv2.KeyPoint_convert(match_lists[1])

			distances = match_lists[2]
			Dmatches = []
			for k in range(len(kp1)):
			# convert the distances to DMatches for use with drawMatches
				Dmatches.append(cv2.DMatch(k,k,distances[k]))

			if System == "linux":
				drawn_matches = drawMatches(images[j],kp2,images[i],kp1,Dmatches)

			if System == "mac":
				drawn_matches = cv2.drawMatches(images[i],kp1,images[j],kp2,Dmatches,None)

			cv2.imshow("matches", drawn_matches)
			cv2.waitKey(0)

	for i in range(len(descriptors)):
		for j in range(i+1, len(descriptors)):
			match_lists = filtered_matches[(i,j)]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[3]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert(match_lists[0])
				kp2 = cv2.KeyPoint_convert(match_lists[3])

			distances = match_lists[2]
			Dmatches = []
			for k in range(len(kp1)):
			# convert the distances to DMatches for use with drawMatches
				Dmatches.append(cv2.DMatch(k,k,distances[k]))

			if System == "linux":
				drawn_matches = drawMatches(images[j],kp2,images[i],kp1,Dmatches)

			if System == "mac":
				drawn_matches = cv2.drawMatches(images[i],kp1,images[j],kp2,Dmatches,None)

			cv2.imshow("matches", drawn_matches)
			cv2.waitKey(0)

	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

	used_images = [0]*len(images)
	best_score = 0
	best_pair = (0,0)
	for image_pair in filtered_matches:
		score = filtered_matches[image_pair][5]
		print(score)
		if score > best_score:
			best_score = score
			best_pair = image_pair

	image_a = images[best_pair[0]]
	image_b = images[best_pair[1]]
	image_a_color = colored_images[best_pair[0]]
	image_b_color = colored_images[best_pair[1]]
	h = filtered_matches[best_pair][4]

	warp = warpTwoImages(image_a, image_b, h)

	color_warp = warpTwoImages(image_a_color, image_b_color, h)
	cv2.imshow("stitch_warp", color_warp)
	cv2.waitKey(0)

	used_images[best_pair[0]] = 1
	used_images[best_pair[1]] = 1

	img32_warp = np.float32(warp)
	dst_warp = cv2.cornerHarris(img32_warp, 2,3,0.04)

	dst_warp[dst_warp<1000000] = 0

	cv2.imshow("d", img32_warp)
	cv2.waitKey(0)

	cv2.imshow("d", dst_warp)
	cv2.waitKey(0)

	amns_warp = []
	n_best = NumFeatures
	ris_warp = []
	best_corner_warp = np.zeros(dst_warp.shape)

	x_max = best_corner_warp.shape[0]
	y_max = best_corner_warp.shape[1]
	for j in range(20,x_max-20):
		for k in range(20,y_max-20):
			if dst_warp[j][k] > 0.0:
#				print((j,k,corners[i][j][k]))
				amns_warp.append((j,k,dst_warp[j][k]))

	n_strong = len(amns_warp)
	for j in range(n_strong):
		ri = 10e+20
		for k in range(n_strong):
			distance = 10e+20
			if amns_warp[k][2] > amns_warp[j][2]:
				distance = ((amns_warp[k][0]-amns_warp[j][0])**2 + (amns_warp[k][1]-amns_warp[j][1])**2)
			if distance < ri:
				ri = distance
		ris_warp.append((amns_warp[j][0],amns_warp[j][1],ri))

	ris_warp = sorted(ris_warp, key = lambda x: x[2], reverse=True)
	ris_warp = ris_warp[:n_best]

	for ri in ris_warp:
		best_corner_warp[ri[0]][ri[1]] = ri[2]

	cv2.imshow("best_corner_warp", best_corner_warp)
	cv2.waitKey(0)


	descriptors_warp = []

	for j in range(len(ris_warp)):
		image = warp
		good_corner = ris_warp[j]
		y = good_corner[0]
		x = good_corner[1]
		desc_matrix = image[y-20:y+20,x-20:x+20]

		blur = cv2.GaussianBlur(desc_matrix, (5,5), 1)

		sub = cv2.resize(blur, (8,8))

		flat = sub.flatten()

		mean = np.mean(flat)
		mean_0 = flat - mean

		std = np.std(mean_0)
		std_1 = mean_0/std

		descriptors_warp.append(((x,y),std_1))


	matches_warp = {}
	threshold = .5

	print("got descriptors")

	for j in range(len(used_images)):
		if used_images[j] == 0:
			# each pair of images will have 3 lists where the first list is a list
			# of points in image 1, the second is a list of corresponding matching
			# points in image 2 and the third is a list of corresponding distances of that match
			matches_warp.update({j:[[],[],[]]})
			for m in range(len(descriptors_warp)):
				best_distance = 5000
				best_point = None
				second_best_distance = 10000
				second_best_point = None
				# for each corner m in image i, look at each corner n in image j
				# if the distance of those descriptors is less than m's distance
				# to any other n, save that distance and n
				for n in range(len(descriptors[j])):
					if (descriptors_warp[m][0] not in matches_warp[j][0] and \
					descriptors[j][n][0] not in matches_warp[j][1]):
						difference = descriptors_warp[m][1] - descriptors[j][n][1]
						square = np.square(difference)
						distance = np.sum(square)
						if distance < best_distance:
							second_best_distance = best_distance
							second_best_point = best_point
							best_distance = distance
							best_point = n

				# for each corner m in image i, if the ratio of its distance
				# to the best corner n of image j is less than a threshold,
				# save those points as a match
				if best_distance/second_best_distance < threshold:
					matches_warp[j][0].append(descriptors_warp[m][0])
					matches_warp[j][1].append(descriptors[j][best_point][0])
					matches_warp[j][2].append(best_distance)

	for j in range(len(used_images)):
		if used_images[j] == 0:
			match_lists = matches_warp[j]
	#		match_lists[0] = [(x,y) for (y,x) in match_lists[0]]
	#		match_lists[1] = [(x,y) for (y,x) in match_lists[1]]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[1]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert(match_lists[0])
				kp2 = cv2.KeyPoint_convert(match_lists[1])

			distances = match_lists[2]
			Dmatches = []
			for k in range(len(kp1)):
				# convert the distances to DMatches for use with drawMatches
				Dmatches.append(cv2.DMatch(k,k,distances[k]))

			if System == "linux":
				drawn_matches = drawMatches(images[j],kp2,images[i],kp1,Dmatches)

			if System == "mac":
				drawn_matches = cv2.drawMatches(images[i],kp1,images[j],kp2,Dmatches,None)

			cv2.imshow("matches", drawn_matches)
			cv2.waitKey(0)

	n_max = 1000
	tau = 100
	threshold = .95

	filtered_matches_warp = {}

	for q in range(len(used_images)):
		if used_images[q] == 0:
			filtered_matches_warp.update({q:[[],[],[],[],[],[]]})

			best_h = np.zeros((3,3))
			best_in_num = 0
			best_inliers_1 = []
			best_inliers_2 = []
			best_distances = []
			best_inliers_prime = []
			best_score = 0

			num_matches = len(matches_warp[q][0])
			im1_points = matches_warp[q][0]
			im2_points = matches_warp[q][1]
			im_distances = matches_warp[q][2]
			for n in range(n_max):
				if num_matches < 4:
					continue
				test_points = random.sample(range(num_matches),4)
				im1_test = [im1_points[num] for num in test_points]
				im2_test = [im2_points[num] for num in test_points]

				im1_array = np.zeros((4,2),dtype=np.float32)
				im2_array = np.zeros((4,2),dtype=np.float32)

				for point in range(4):
					#im1_array[point] = im1_test[point][::-1]
					#im2_array[point] = im2_test[point][::-1]
					for coord in range(2):
						im1_array[point][coord] = im1_test[point][coord]
						im2_array[point][coord] = im2_test[point][coord]

				h = cv2.getPerspectiveTransform(im1_array, im2_array)

				in_num = 0
				inliers_1 = []
				inliers_2 = []
				distances = []
				inliers_prime = []
				for i in range(len(im1_points)):
					pi = np.array([[im1_points[i][0]],[im1_points[i][1]],[1]],dtype=np.float32)
					pi_prime = np.array([[im2_points[i][0]],[im2_points[i][1]],[1]],dtype=np.float32)
					hpi = np.dot(h,pi)
					difference = np.subtract(hpi,pi_prime)
					square = np.square(difference)
					sum = np.sum(square)
					if sum < tau:
						in_num += 1
						inliers_1.append(im1_points[i])
						inliers_2.append(im2_points[i])
						distances.append(im_distances[i])
						inliers_prime.append((hpi.item(0),hpi.item(1)))

				if in_num > best_in_num:
					best_h = h
					best_in_num = in_num
					best_inliers_1 = inliers_1
					best_inliers_2 = inliers_2
					best_distances = distances
					best_inliers_prime = inliers_prime
					best_score = best_in_num/len(im1_points)
					if best_score > threshold:
						break

			new_h = np.zeros((3,3))
			for i in range(100):
				if best_in_num < 4:
					break
				test_points = random.sample(range(best_in_num),4)
				im1_test = [best_inliers_1[num] for num in test_points]
				im2_test = [best_inliers_2[num] for num in test_points]

				im1_array = np.zeros((4,2),dtype=np.float32)
				im2_array = np.zeros((4,2),dtype=np.float32)

				for point in range(4):
					for coord in range(2):
						im1_array[point][coord] = im1_test[point][coord]
						im2_array[point][coord] = im2_test[point][coord]

				h = cv2.getPerspectiveTransform(im1_array, im2_array)

				new_h = np.add(new_h,h)

			new_h = new_h * 0.01

			best_inliers_prime = []
			for i in range(len(best_inliers_1)):
				pi = np.array([[best_inliers_1[i][0]],[best_inliers_1[i][1]],[1]],dtype=np.float32)
				pi_prime = np.array([[best_inliers_2[i][0]],[best_inliers_2[i][1]],[1]],dtype=np.float32)
				hpi = np.dot(best_h,pi)
				best_inliers_prime.append((hpi.item(0),hpi.item(1)))

			filtered_matches_warp[q][0] = best_inliers_1
			filtered_matches_warp[q][1] = best_inliers_2
			filtered_matches_warp[q][2] = best_distances
			filtered_matches_warp[q][3] = best_inliers_prime
			filtered_matches_warp[q][4] = best_h
			filtered_matches_warp[q][5] = best_score

	print(used_images)
	for j in range(i+1, len(descriptors)):
		if used_images[j] == 0:
			match_lists = filtered_matches_warp[j]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[1]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert(match_lists[0])
				kp2 = cv2.KeyPoint_convert(match_lists[1])

			distances = match_lists[2]
			Dmatches = []
			for k in range(len(kp1)):
			# convert the distances to DMatches for use with drawMatches
				Dmatches.append(cv2.DMatch(k,k,distances[k]))

			if System == "linux":
				drawn_matches = drawMatches(images[j],kp2,images[i],kp1,Dmatches)

			if System == "mac":
				drawn_matches = cv2.drawMatches(images[i],kp1,images[j],kp2,Dmatches,None)

			cv2.imshow("matches", drawn_matches)
			cv2.waitKey(0)

	for j in range(len(used_images)):
		if used_images[j] == 0:
			match_lists = filtered_matches_warp[j]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[0], point[1], 10) for point in match_lists[3]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert(match_lists[0])
				kp2 = cv2.KeyPoint_convert(match_lists[3])

			distances = match_lists[2]
			Dmatches = []
			for k in range(len(kp1)):
			# convert the distances to DMatches for use with drawMatches
				Dmatches.append(cv2.DMatch(k,k,distances[k]))

			if System == "linux":
				drawn_matches = drawMatches(images[j],kp2,images[i],kp1,Dmatches)

			if System == "mac":
				drawn_matches = cv2.drawMatches(images[i],kp1,images[j],kp2,Dmatches,None)

			cv2.imshow("matches", drawn_matches)
			cv2.waitKey(0)

	best_score = -10
	best_num = -1
	for image_num in filtered_matches_warp:
		print(image_num)
		score = filtered_matches_warp[image_num][5]
		print(score)
		if score > best_score:
			best_score = score
			best_num = image_num

	image_a = warp
	image_b = images[best_num]
	image_a_color = color_warp
	image_b_color = colored_images[best_num]
	h = filtered_matches_warp[best_num][4]

	warp = warpTwoImages(image_a, image_b, h)

	color_warp = warpTwoImages(image_a_color, image_b_color, h)
	cv2.imshow("stitch_warp_final", color_warp)
	cv2.waitKey(0)


	# current_warped_image = None
	# pair_scores = {}
	#
	#
	# for length of images -1:
	# 	find highest score image
	# 	stitch them
	# 	make that the current warped image
	# 	get its corners
	# 	for image in remaining images:
	# 		match it with the rest of the images
	# 		store the scores



#			min_pixel = np.float32([[0],[0],[1]])

#			offset = np.dot(np.linalg.inv(h),min_pixel)
#			x_shift = int(offset[0][0] * -1)
#			y_shift = int(offset[1][0] * -1)

#			print(x_shift)
#			print(y_shift)

#			warp = cv2.warpPerspective(image_a, np.linalg.inv(h), (700,900))
#			print(warp)
#			cv2.imshow("stitch", warp)
#			cv2.waitKey(0)

#			cv2.imshow("image_b", image_b)
#			cv2.waitKey(0)

	#		T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

	#		warp = cv2.warpAffine(warp, T, (700,900))
	#		cv2.imshow("stitch", warp)
	#		cv2.waitKey(0)

	#		warp[0:image_b.shape[0],0:image_b.shape[1]] = image_b

	#		cv2.imshow("stitch", warp)
	#		cv2.waitKey(0)



if __name__ == '__main__':
	main()
