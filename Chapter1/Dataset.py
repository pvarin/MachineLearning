import numpy as np
import gzip, struct

def parseMNIST(dataPath, labelsPath):
	dataFile = gzip.GzipFile(dataPath)
	labelFile = gzip.GzipFile(labelsPath)
	
	# push out the first 4 useless bytes
	struct.unpack('>i',dataFile.read(4))[0]
	numImgs, width, height = struct.unpack('>III',dataFile.read(12))

	# push out the first 8 bytes of the labels file
	struct.unpack('>II',labelFile.read(8))

	# print useful output
	print "Loading %s images, each %s by %s" % (numImgs,width,height)

	# allocate memory 
	labels = np.zeros(numImgs)
	data = np.zeros((numImgs, width*height))

	# load data and labels
	for i in xrange(numImgs):
		labels[i] = struct.unpack('B',labelFile.read(1))[0]
		d = dataFile.read(width*height)
		data[i,:] = np.array(struct.unpack('B'*width*height,d))

	print 'Done'
	return data, labels

class Dataset(object):
	def loadMNIST(self, dataPath, labelsPath):
		self.data, self.labels = parseMNIST(dataPath,labelsPath)
		self.maxLabel = np.nanmax(self.labels)

class Classifier(Dataset):
	def loadMNIST(self, dataPath, labelsPath, test=False):
		if test:
			self.testData, self.testLabels = parseMNIST(dataPath,labelsPath)
		else:
			self.data, self.labels = parseMNIST(dataPath,labelsPath)
			self.maxLabel = np.nanmax(self.labels)
	
	def classifyRate(self,*args, **kwargs):
		labels = self.classify(self.testData, *args, **kwargs)
		print labels
		print self.testLabels
		numCorrect = sum(labels == self.testLabels)
		return float(numCorrect)/len(labels)

	def classify(self,data):
		raise NotImplementedError

if __name__ == '__main__':
	# Test the Dataset class
	a = Dataset()
	a.loadMNIST('../data/MNIST_Training_Data.gz','../data/MNIST_Training_Labels.gz')

	# Test the Classifier class
	a = Classifier()
	a.loadMNIST('../data/MNIST_Training_Data.gz','../data/MNIST_Training_Labels.gz')
	a.loadMNIST('../data/MNIST_Test_Data.gz','../data/MNIST_Test_Labels.gz', test=True)