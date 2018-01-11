import sys,os,itertools,time,glob,cv2
from sklearn.utils import shuffle
import numpy as np
def drawProgressBar(percent, barLen = 50):
	sys.stdout.write("\r")
	progress = ""
	for i in range(barLen):
		if i<int(barLen * percent):
			progress += "="
		else:
			progress += " "
	sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
	sys.stdout.flush()

def load_train(train_path):
	images = []
	#img_names = []
	classes = []
	path = train_path
	file_names = os.listdir(os.path.join(os.getcwd(),train_path))
	counter = 1
	print("Creating Classes, reading images and breaking things ...\n")
	for file in file_names:
		drawProgressBar(counter/len(file_names))
		#print(file)
		classes.append(file.split("_")[0])
		image = cv2.imread(os.path.join(os.getcwd(),train_path,file))
		#image = cv2.imread(str(file))
		#print("\ntype of img",type(image),"\t type of temp",type(temp))
		image = image.astype(np.float32)
		image = np.multiply(image, 1.0/255.0) #normalizing the pixel intensities
		images.append(image)
		#img_names.append(os.path.basename(file))
		counter += 1
	print("\nDone!")
	images = np.array(images)	
	#classes now has all the labels. order preserved
	#but we need the classes to be floats/ints so lets map the shit out of them
	d = {ni:indi for indi, ni in enumerate(set(classes))}
	classes = [d[ni] for ni in classes]
	classes = np.array(classes)
	n_values = np.max(classes)+1
	classes = np.eye(n_values)[classes]
	return (images,classes)
#load_train(os.path.join(os.getcwd(),"im_train"),12)

class DataSet(object):

	def __init__(self, images, labels):
		self._num_examples = images.shape[0]
		self._images = images
		self._labels = labels
		#self._img_names = img_names
		#self._cls = cls
		self._epochs_done = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	#@property
	#def img_names(self):
	#	return self._img_names

	#@property
	#def cls(self):
	#	return self._cls

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_done(self):
		return self._epochs_done

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:
		  # After each epoch we update this
		  self._epochs_done += 1
		  start = 0
		  self._index_in_epoch = batch_size
		  assert batch_size <= self._num_examples
		end = self._index_in_epoch

		return self._images[start:end], self._labels[start:end]#, self._cls[start:end]

def read_train_sets(train_path, validation_size,image_size = 28):
	class DataSets(object):
		pass
	data_sets = DataSets()

	images,  cls = load_train(train_path)
	#labels = cls
	#labels = np.array(labels)
	images, labels = shuffle(images, cls)  
	#print("shape of labels, cls",labels.shape,cls.shape)
	if isinstance(validation_size, float):
		validation_size = int(validation_size * images.shape[0])

	validation_images = images[:validation_size]
	validation_labels = labels[:validation_size]
	#validation_img_names = img_names[:validation_size]
	#validation_cls = cls[:validation_size]

	train_images = images[validation_size:]
	train_labels = labels[validation_size:]
	#train_img_names = img_names[validation_size:]
	#train_cls = cls[validation_size:]

	data_sets.train = DataSet(train_images, train_labels)
	data_sets.valid = DataSet(validation_images, validation_labels)

	return data_sets,labels
