import sys
sys.path.append('..')
import Dataset
import numpy as np
import scipy.spatial.distance as dist

class KNNDataset(Dataset.Classifier):
	def classify(self, data, k=1):
		labels = np.zeros(len(data))

		for i,d in enumerate(data):
			print "%s of %s" % (i, len(data))
			# allocate space for label counts and calculate distance
			lCount = np.zeros(self.maxLabel+1)
			distance = dist.cdist(self.data,np.array([d]))

			# find the labels of the k nearest 
			for j in xrange(k):
				idx = np.argmin(distance)
				distance[idx] = np.inf
				lCount[self.labels[idx]] += 1
			labels[i] = np.argmax(lCount)
		
		return labels



if __name__=='__main__':
	k = KNNDataset()
	k.loadMNIST('../data/MNIST_Training_Data.gz','../data/MNIST_Training_Labels.gz')
	k.loadMNIST('../data/MNIST_Test_Data.gz','../data/MNIST_Test_Labels.gz', test=True)
	print k.classifyRate()