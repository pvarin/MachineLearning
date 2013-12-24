import numpy as np
import gzip, struct

class Dataset(object):
	def loadMNIST(self, dataPath, labelsPath):
		self.dataFile = gzip.GzipFile(dataPath)
		self.labelFile = gzip.GzipFile(labelsPath)
		
		# push out the first 4 useless bytes
		struct.unpack('>i',self.dataFile.read(4))[0]
		numImgs, width, height = struct.unpack('>III',self.dataFile.read(12))

		# push out the first 8 bytes of the labels file
		struct.unpack('>II',self.labelFile.read(8))

		# print useful output
		print "Loading %s images, each %s by %s" % (numImgs,width,height)

		# # allocate memory 
		self.labels = np.zeros(numImgs)
		self.data = np.zeros((numImgs, width*height))

		# load data and labels
		for i in xrange(numImgs):
			self.labels[i] = struct.unpack('B',self.labelFile.read(1))[0]
			d = self.dataFile.read(width*height)
			self.data[i,:] = np.array(struct.unpack('B'*width*height,d))

		print 'Done'

if __name__ == '__main__':
	# Test the MNISTDataset class
	a = Dataset()
	a.loadMNIST('../data/MNIST_Test_Data.gz','../data/MNIST_Test_Labels.gz')
