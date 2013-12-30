import numpy as np
import gzip, struct

class Dataset(object):

	def save(self, dataPath, labelsPath):
		np.save(dataPath, self.data)
		np.save(labelsPath, self.labels)

	def load(self, dataPath, labelsPath):
		self.data = np.load(dataPath)
		self.labels = np.load(labelsPath)	


class Classifier(Dataset):
	def load(self, dataPath, labelsPath, test=False):
		if test:
			self.testData = np.load(dataPath)
			self.testLabels = np.load(labelsPath)
		else:
			self.data = np.load(dataPath)
			self.labels = np.load(labelsPath)
			self.maxLabel = np.nanmax(self.labels)
	
	def classifyRate(self,*args, **kwargs):
		labels = self.classify(self.testData, *args, **kwargs)
		numCorrect = sum(labels == self.testLabels)
		return float(numCorrect)/len(labels)

	def classify(self,data):
		raise NotImplementedError

if __name__ == '__main__':
	# Test the Dataset class
	a = Dataset()
	print "Loading MNIST_Training_Data"
	a.load('../data/MNIST_Test_Data.npy','../data/MNIST_Test_Labels.npy')
	print "Saving locally"
	a.save('MNIST_Training_Data.npy','MNIST_Training_Labels.npy')

	# Test the Classifier class
	a = Classifier()
	print "Loading Training Data"
	a.load('../data/MNIST_Training_Data.npy','../data/MNIST_Training_Labels.npy')
	print "Loading Test Data"
	a.load('../data/MNIST_Test_Data.npy','../data/MNIST_Test_Labels.npy', test=True)