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
# Add any python libraries here



def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	
	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageSetBasePath', default="../Data/Train/Set1", help='Number of best features to extract from each image, Default: ../Data/Train/Set1')
	Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	ImageSetBasePath = Args.ImageSetBasePath


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

		dst[dst<100000] = 0

		print(dst + img)

		print(img)
		print(img)

		cv2.imshow("d", img)
		cv2.waitKey(0)
		print(dst)
		cv2.imshow("d", dst)
		cv2.waitKey(0)


	for corner in corners:
		print(corner)
		cv2.imshow("d", corner)
		cv2.waitKey(0)




	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""


	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

if __name__ == '__main__':
	main()
 
