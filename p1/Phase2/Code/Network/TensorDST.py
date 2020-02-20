from __future__ import print_function
import tensorflow as tf
import sys
import cv2
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def TensorDST(H4Pt, OriginalCorners, Perturbation, MiniBatchSize):

	PredictedCorners = H4Pt * Perturbation + OriginalCorners


	# Uncomment to get homography from predicted to original 

	#temp = PredictedCorners
	#PredictedCorners = OriginalCorners
	#OriginalCorners = temp


	A = []

	for i in range(4):


		A_i = tf.concat([

		tf.stack([tf.transpose([OriginalCorners[:,2*i]]), tf.transpose([OriginalCorners[:,2*i+1]]), tf.transpose([tf.ones(MiniBatchSize)]), 
			tf.transpose([tf.zeros(MiniBatchSize)]), tf.transpose([tf.zeros(MiniBatchSize)]),tf.transpose([tf.zeros(MiniBatchSize)]), 
			-1 * tf.transpose([OriginalCorners[:,2*i]]) * tf.transpose([PredictedCorners[:,2*i]]), -1 * tf.transpose([OriginalCorners[:,2*i+1]]) * tf.transpose([PredictedCorners[:,2*i]])], axis=1),

		tf.stack([tf.transpose([tf.zeros(MiniBatchSize)]),tf.transpose([tf.zeros(MiniBatchSize)]) ,tf.transpose([tf.zeros(MiniBatchSize)]) , 
			tf.transpose([OriginalCorners[:,2*i]]), tf.transpose([OriginalCorners[:,2*i+1]]), tf.transpose([tf.ones(MiniBatchSize)]), 
			-1 * tf.transpose([OriginalCorners[:,2*i]]) * tf.transpose([PredictedCorners[:,2*i+1]]), -1 * tf.transpose([OriginalCorners[:,2*i+1]]) * tf.transpose([PredictedCorners[:,2*i+1]])], axis=1)], axis = 2)

		A.append(tf.transpose(A_i, perm = (0,2,1)))

	A = tf.concat(A,axis= 1)


	b = PredictedCorners

	h = tf.transpose(tf.matmul(tf.linalg.pinv(A), b[:,:,None]),perm = (0,2,1))[:,0,:]

	h = tf.concat([h, tf.transpose([tf.ones(MiniBatchSize)])], axis = 1)

	#print(tf.reshape(h,(-1,3,3)))

	#print(PredictedCorners[0])
	#print(OriginalCorners[0])

	#pred_points = tf.reshape(PredictedCorners[0], (4,2))
	#orig_points = tf.reshape(OriginalCorners[0], (4,2))

	#print(pred_points)
	#print(orig_points.astype(tf.float32))

	#print(cv2.getPerspectiveTransform(pred_points.astype(tf.float32), orig_points.astype(tf.float32)))
	#f = cv2.getPerspectiveTransform(orig_points.astype(tf.float32), pred_points.astype(tf.float32))
	#print(f)

	return h





def main():
	H4Pt = tf.array([[0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4], tf.array([0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4]) * 2])
	Perturbation = 10

	OriginalCorners = tf.array([[5,10,1,2,5,10,1,2], tf.array([5,10,1,2,5,10,1,2])*2])

	TensorDST(H4Pt, OriginalCorners, Perturbation, 2)




		
	
if __name__ == '__main__':
	main()


