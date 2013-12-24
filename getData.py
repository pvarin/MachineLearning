import urllib, os

dataDir = 'data'

try:
	os.mkdir(dataDir)
except OSError:
	pass

# MNIST Data
data = [('MNIST_Training_Data','http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'),
		('MNIST_Training_Labels','http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'),
		('MNIST_Test_Data','http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'),
		('MNIST_Test_Labels','http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')]

for filename, url in data:
	path = os.path.join(dataDir, filename + '.' + url.split('.')[-1])

	# check if the path already exists
	if os.path.exists(path):
		print "skipping %-25s file already exists" % filename
	else:
		# download data
		print "Downloading %s to %s..." % (filename, path)
		urllib.urlretrieve(url,path)
		print "%s successfully downloaded" % filename

print "\nAll data successfully downloaded, stored in directory: %s" % (dataDir)