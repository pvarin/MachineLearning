import sys
sys.path.append('..')
import Dataset
import numpy as np
import scipy.spatial.distance as dist

class KNNDataset(Dataset.Classifier):
	def classify(self, data, k=1, verbose=True):
		labels = np.zeros(len(data))

		for i,d in enumerate(data):
			if verbose:
				print "%s of %s" % (i, len(data))
			# allocate space for label counts and calculate distance
			lCount = np.zeros(self.maxLabel+1)
			distance = dist.cdist(self.data,np.array([d]))

			# find the labels of the k nearest 
			for j in xrange(k):
				idx = np.nanargmin(distance)
				distance[idx] = np.inf
				lCount[self.labels[idx]] += 1
			labels[i] = np.argmax(lCount)
		
		return labels

class CNNDataset(KNNDataset):
	def loadMNIST(self, dataPath, labelsPath, test=False, verbose=True):
		# load data as previously
		super(CNNDataset,self).loadMNIST(dataPath,labelsPath,test=test)
		
		# don't compress the test data
		if test:
			return

		# compress the training data
		if verbose:
			print "Filtering data for CNN"
			
		for i in xrange(len(self.data)):
			if verbose:
				print "%s of %s" % (i+1, len(self.data))
			temp = self.data[i].copy()
			self.data[i] = np.nan
			if self.classify(temp[:,np.newaxis].T,verbose=False) != self.labels[i]:
				self.data[i] = temp

		keepIdx = np.isfinite(self.data)
		self.data = self.data[keepIdx]
		self.labels = self.labels[keepIdx]



if __name__=='__main__':
	# # Test KNN
	# k = KNNDataset()
	# k.loadMNIST('../data/MNIST_Training_Data.gz','../data/MNIST_Training_Labels.gz')
	# k.loadMNIST('../data/MNIST_Test_Data.gz','../data/MNIST_Test_Labels.gz', test=True)
	# print k.classifyRate()

	# Test CNN
	k = CNNDataset()
	k.loadMNIST('../data/MNIST_Training_Data.gz','../data/MNIST_Training_Labels.gz')
	k.loadMNIST('../data/MNIST_Test_Data.gz','../data/MNIST_Test_Labels.gz', test=True)
	print k.classifyRate()