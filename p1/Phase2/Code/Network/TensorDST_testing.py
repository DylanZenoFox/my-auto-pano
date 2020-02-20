from __future__ import print_function
import tensorflow as tf
import sys
import cv2
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def TensorDST(H4Pt, OriginalCorners, Perturbation, MiniBatchSize):

	PredictedCorners = H4Pt * Perturbation + OriginalCorners

	#temp = PredictedCorners
	#PredictedCorners = OriginalCorners
	#OriginalCorners = temp


	A = []

	for i in range(4):


		A_i = np.concatenate([

		np.stack([np.transpose([OriginalCorners[:,2*i]]), np.transpose([OriginalCorners[:,2*i+1]]), np.transpose([np.ones(MiniBatchSize)]), 
			np.transpose([np.zeros(MiniBatchSize)]), np.transpose([np.zeros(MiniBatchSize)]),np.transpose([np.zeros(MiniBatchSize)]), 
			-1 * np.transpose([OriginalCorners[:,2*i]]) * np.transpose([PredictedCorners[:,2*i]]), -1 * np.transpose([OriginalCorners[:,2*i+1]]) * np.transpose([PredictedCorners[:,2*i]])], axis=1),

		np.stack([np.transpose([np.zeros(MiniBatchSize)]),np.transpose([np.zeros(MiniBatchSize)]) ,np.transpose([np.zeros(MiniBatchSize)]) , 
			np.transpose([OriginalCorners[:,2*i]]), np.transpose([OriginalCorners[:,2*i+1]]), np.transpose([np.ones(MiniBatchSize)]), 
			-1 * np.transpose([OriginalCorners[:,2*i]]) * np.transpose([PredictedCorners[:,2*i+1]]), -1 * np.transpose([OriginalCorners[:,2*i+1]]) * np.transpose([PredictedCorners[:,2*i+1]])], axis=1)], axis = 2)

		A.append(np.transpose(A_i, axes = (0,2,1)))

	A = np.concatenate(A,axis = 1)


	b = PredictedCorners

	h = np.transpose(np.matmul(np.linalg.pinv(A), b[:,:,None]),axes = (0,2,1))[:,0,:]

	h = np.concatenate([h, np.transpose([np.ones(MiniBatchSize)])], axis = 1)

	print(np.reshape(h,(-1,3,3)))

	#print(PredictedCorners[0])
	#print(OriginalCorners[0])

	pred_points = np.reshape(PredictedCorners[0], (4,2))
	orig_points = np.reshape(OriginalCorners[0], (4,2))

	#print(pred_points)
	#print(orig_points.astype(np.float32))

	#print(cv2.getPerspectiveTransform(pred_points.astype(np.float32), orig_points.astype(np.float32)))
	f = cv2.getPerspectiveTransform(orig_points.astype(np.float32), pred_points.astype(np.float32))
	print(f)




	return h





def main():
	H4Pt = np.array([[0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4], np.array([0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4]) * 2])
	Perturbation = 10

	OriginalCorners = np.array([[5,10,1,2,5,10,1,2], np.array([5,10,1,2,5,10,1,2])*2])

	TensorDST(H4Pt, OriginalCorners, Perturbation, 2)




		
	
if __name__ == '__main__':
	main()


