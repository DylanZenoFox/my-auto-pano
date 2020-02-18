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
import argparse
import os
import random
import matplotlib.pyplot as plt
from Utils import *
# Add any python libraries here



def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageSetBasePath', default="../Data/Train/Set1", help='Number of best features to extract from each image, Default: ../Data/Train/Set1')
	Parser.add_argument('--NumFeatures', default=200,type=int ,help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--System', default="mac", help="Sets system for visualization, Options: 'linux', 'mac'")

	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	ImageSetBasePath = Args.ImageSetBasePath
	System = Args.System


	"""
	Read a set of images for Panorama stitching
	"""

	images = []

	for file in os.listdir(ImageSetBasePath):
		if(file.endswith(".png") or file.endswith(".jpg")):
			file = ImageSetBasePath + "/" +  file
			images.append(cv2.imread(file,cv2.IMREAD_GRAYSCALE))
		else:
			continue


	for corner in images:
		cv2.imshow("d", corner)
		cv2.waitKey(0)


	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	corners = []

	for img in images:
		img32 = np.float32(img)
		dst = cv2.cornerHarris(img32, 2,3,0.04)

		#dst = cv2.dilate(dst, None, iterations=1)

		dst[dst<1000000] = 0

#		print(dst + img)

#		print(img)
#		print(img)

		cv2.imshow("d", img)
		cv2.waitKey(0)
#		print(dst)
		corners.append(dst)
		cv2.imshow("d", dst)
		cv2.waitKey(0)


	for corner in corners:
#		print(corner)
		cv2.imshow("c", corner)
		cv2.waitKey(0)




	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	amns = []
	n_best = NumFeatures
	ris = []
	best_corners = []

	for i in range(len(corners)):
		x_max = corners[i].shape[0]
		y_max = corners[i].shape[1]
		amns.append([])
		for j in range(20,x_max-20):
			for k in range(20,y_max-20):
				if corners[i][j][k] > 0.0:
#					print((j,k,corners[i][j][k]))
					amns[i].append((j,k,corners[i][j][k]))

	for i in range(len(amns)):
		ris.append([])
		n_strong = len(amns[i])
		for j in range(n_strong):
			ri = 10e+20
			for k in range(n_strong):
				distance = 10e+20
				if amns[i][k][2] > amns[i][j][2]:
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
			x = good_corner[0]
			y = good_corner[1]
			desc_matrix = image[x-20:x+20,y-20:y+20]

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
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[1], point[0], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[1], point[0], 10) for point in match_lists[1]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert([(x,y) for (y,x) in match_lists[0]])
				kp2 = cv2.KeyPoint_convert([(x,y) for (y,x) in match_lists[1]])

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

	for image_pair in matches:
		filtered_matches.update({image_pair:[[],[],[],[],[]]})

		best_h = np.zeros((3,3))
		best_in_num = 0
		best_inliers_1 = []
		best_inliers_2 = []
		best_distances = []
		best_inliers_prime = []

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
				if best_in_num/len(im1_points) > threshold:
					break

		filtered_matches[image_pair][0] = best_inliers_1
		filtered_matches[image_pair][1] = best_inliers_2
		filtered_matches[image_pair][2] = best_distances
		filtered_matches[image_pair][3] = best_inliers_prime
		filtered_matches[image_pair][4] = best_h

	for i in range(len(descriptors)):
		for j in range(i+1, len(descriptors)):
			match_lists = filtered_matches[(i,j)]
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[1], point[0], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[1], point[0], 10) for point in match_lists[1]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert([(x,y) for (y,x) in match_lists[0]])
				kp2 = cv2.KeyPoint_convert([(x,y) for (y,x) in match_lists[1]])

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
			print(match_lists[3])
			# convert the points into KeyPoints for use with drawMatches

			kp1 = []
			kp2 = []

			if System == "linux":
				kp1 = [cv2.KeyPoint(point[1], point[0], 10) for point in match_lists[0]]
				kp2 = [cv2.KeyPoint(point[1], point[0], 10) for point in match_lists[3]]

			if System == "mac":
				kp1 = cv2.KeyPoint_convert([(x,y) for (y,x) in match_lists[0]])
				kp2 = cv2.KeyPoint_convert([(x,y) for (y,x) in match_lists[3]])

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

	for i in range(len(descriptors)):
		for j in range(i+1, len(descriptors)):
			match_lists = filtered_matches[(i,j)]
			image_a = images[i]
			image_b = images[j]
			h = match_lists[4]
			print(h)

			min_pixel = np.float32([[0],[0],[1]])

			offset = np.dot(np.linalg.inv(h),min_pixel)
			x_shift = int(offset[0][0] * -1)
			y_shift = int(offset[1][0] * -1)

			print(x_shift)
			print(y_shift)

			warp = cv2.warpPerspective(image_a, np.linalg.inv(h), (700,900))
			print(warp)
			cv2.imshow("stich", warp)
			cv2.waitKey(0)

			T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

			warp = cv2.warpAffine(warp, T, (700,900))
			cv2.imshow("stitch", warp)
			cv2.waitKey(0)

			warp[0:image_b.shape[0],0:image_b.shape[1]] = image_b

			cv2.imshow("stich", warp)
			cv2.waitKey(0)



if __name__ == '__main__':
	main()
