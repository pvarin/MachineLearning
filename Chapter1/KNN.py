import sys
sys.path.append('..')
import Dataset
import numpy as np
from scipy import spatial

class KNNDataset(Dataset.Classifier):

	def load(self, *args, **kwargs):
		super(KNNDataset,self).load(*args,**kwargs)

		if not kwargs.get('test',False):
			print 'Constructing K-D Tree'
			self.tree = spatial.cKDTree(self.data)

	def classify(self, data, k=1):
		# Allocate memory
		classifiedLabels = np.zeros(len(data))

		# Find Nearest-Neighbors
		_, idx = self.tree.query(data,k=k)
		labels = self.labels[idx]

		if k is 1:
			return labels
		else:
			return numpy.amax(labels,axis=1)

class CNNDataset(KNNDataset):
	def load(self, dataPath, labelsPath, test=False, verbose=True):
		# load data as previously
		super(CNNDataset,self).load(dataPath,labelsPath,test=test)
		
		# don't compress the test data
		if test:
			return

		# create the compressed search data
		remIdx = [0]
		it = 0
		self.tree = spatial.cKDTree(self.data[remIdx])
		changed = True
		"Iteratively updating the search tree"
		while len(remIdx) > 0:
			# print iteration number
			print "Iteration %s" % it
			it +=1

			# add data to the search tree
			changed = False
			for i,d in enumerate(self.data):
				if i in remIdx:
					continue
				temp = self.classify(d)
				print i, len(remIdx)
				if temp != self.labels[i]:
					remIdx.append(i)
					if len(remIdx) % 10 == 0:
						self.tree = spatial.cKDTree(self.data[remIdx])
					changed = True

if __name__=='__main__':
	# Test KNN
	# k = KNNDataset()
	# print 'Loading Training Data'
	# k.load('../data/MNIST_Training_Data.npy','../data/MNIST_Training_Labels.npy')
	# print 'Loading Test Data'
	# k.load('../data/MNIST_Test_Data.npy','../data/MNIST_Test_Labels.npy', test=True)
	# print 'Classifying Data'
	# print k.classifyRate()

	# # Test CNN
	k = CNNDataset()
	print 'Loading Training Data'
	k.load('../data/MNIST_Training_Data.npy','../data/MNIST_Training_Labels.npy')
	print 'Loading Test Data'
	k.load('../data/MNIST_Test_Data.npy','../data/MNIST_Test_Labels.npy', test=True)
	print 'Classifying Data'
	print k.classifyRate()
	# k.load('../data/MNIST_Training_Data.npy','../data/MNIST_Training_Labels.npy', test=True)
	# k.load('../data/MNIST_Test_Data.npy','../data/MNIST_Test_Labels.npy', test=False)
	# print k.classifyRate(verbose=False)