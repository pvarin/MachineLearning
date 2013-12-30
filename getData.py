import urllib, os, gzip, struct
import numpy as np

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

if __name__ == "__main__":
	dataDir = 'data'

	try:
		os.mkdir(dataDir)
	except OSError:
		pass

	## Download all data
	# MNIST Data
	data = [('MNIST_Training_Data','http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'),
			('MNIST_Training_Labels','http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'),
			('MNIST_Test_Data','http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'),
			('MNIST_Test_Labels','http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')]

	for filename, url in data:
		path = os.path.join(dataDir, filename + '_Raw.' + url.split('.')[-1])

		# check if the path already exists
		if os.path.exists(path):
			print "skipping %-25s file already exists" % filename
		else:
			# download data
			print "Downloading %s to %s..." % (filename, path)
			urllib.urlretrieve(url,path)
			print "%s successfully downloaded" % filename

	print "All data successfully downloaded, stored in directory: %s" % (dataDir)

	print "\nUnpacking MNIST data to Numpy arrays"
	MNISTData = [("MNIST_Training_Data","MNIST_Training_Labels"),("MNIST_Test_Data","MNIST_Test_Labels")]
	for d,l in MNISTData:
		# determine the paths
		d_path_in = os.path.join(dataDir, d + '_Raw.gz')
		d_path_out = os.path.join(dataDir, d)
		l_path_in = os.path.join(dataDir, l + '_Raw.gz')
		l_path_out = os.path.join(dataDir, l)

		# don't do extra work
		if os.path.exists(d_path_out) and os.path.exists(l_path_out):
			print "Skipping %-19s and %-21s files already exist" % (d,l)
			continue
		print "Unpacking %s and %s" % (d,l)
		# unpack the data
		npData, npLabels = parseMNIST(d_path_in,l_path_in)
		print "saving data to %s" % d_path_out
		np.save(d_path_out,npData)
		print "saving labels to %s" % l_path_out
		np.save(l_path_out,npLabels)
	print "All data successfully unpacked, stored in directory: %s" % (dataDir)