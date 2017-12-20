"""
Author - Sanjay Krishna Gouda
Email -  sgouad@ucsc.edu
architecture and some results in Results.txt
"""
# coding: utf-8

# In[1]:


import tensorflow as tf
import time,sys,os,math,random,itertools,glob,cv2
from datetime import timedelta
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split,ShuffleSplit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#get_ipython().run_line_magic('matplotlib', 'inline')
set_random_seed(2)
#final_project_path = r"M:\Course stuff\Fall 17\CMPS 242\final project"
#os.chdir(final_project_path)



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



imp_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
def load_train(train_path):
	images = []
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
		image = image.astype(np.float32)
		image = np.multiply(image, 1.0/255.0) #normalizing the pixel intensities
		images.append(image)
		counter += 1
	print("\nDone!")
	images = np.array(images)
	#classes now has all the labels. order preserved
	#but we need the classes to be floats/ints so lets map the shit out of them
	for i in range(len(classes)):
		if classes[i] not in imp_labels:
			classes[i] = 'unkown'
	d = {ni:indi for indi, ni in enumerate(set(classes))}
	classes = [d[ni] for ni in classes]
	classes = np.array(classes)
	n_values = np.max(classes)+1
	classes = np.eye(n_values)[classes]
	#classes = np.eye(n_values)[classes.reshape(-1)]
	print("\nDone!")
	print("\n images shape: {}, labels shape: {}".format(images.shape,classes.shape))
	return (images,classes)#(train_x,train_y,test_x,test_y)
	

def split_data(images, labels,test_size = 0.2, random_state = 7, shuffle = False):
	return(train_test_split(images,labels,test_size = test_size,random_state = random_state,shuffle = shuffle))


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

	m = X.shape[0]                  # number of training examples
	mini_batches = []
	np.random.seed(seed)
	
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches


def create_placeholders(n_H0, n_W0, n_C0, n_y):

	X = tf.placeholder(shape = [None, n_H0, n_W0, n_C0],dtype = tf.float32)
	Y = tf.placeholder(shape = [None, n_y],dtype = tf.float32)

	return X, Y


def initialize_parameters():

	tf.set_random_seed(1)                              # so that your "random" numbers match ours
		

	W1 = tf.get_variable("W1",[7, 7, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W2 = tf.get_variable("W2",[5, 5, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W3 = tf.get_variable("W3",[3, 3, 16, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W4 = tf.get_variable("W4",[2, 2, 8, 4], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W5 = tf.get_variable("W5",[2, 2, 4, 4], initializer = tf.contrib.layers.xavier_initializer(seed = 0))


	parameters = {"W1": W1,
				  "W2": W2,
				  "W3":W3,
				  "W4":W4,
				  "W5":W5}
	
	return parameters

def forward_propagation(X, parameters):

	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']
	W5 = parameters['W5']
	with tf.device('/device:GPU:0'):

		Z1 = tf.nn.conv2d(X,W1,strides = [1,2,2,1], padding = 'SAME')
		A1 = tf.nn.elu(Z1)		
		P1 = tf.nn.max_pool(A1,ksize = [1, 8, 8,1], strides = [1,8,8,1],padding = 'SAME')
		

		Z2 = tf.nn.conv2d(P1, W2, strides=[1,2,2, 1], padding='SAME')
		A2 = tf.nn.elu(Z2)#relu(Z2)
		P2 = tf.nn.max_pool(A2, ksize = [1, 4,4,1], strides = [1, 2,2, 1], padding='SAME')
		
		Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
		A3 = tf.nn.elu(Z3)
		P3 = tf.nn.max_pool(A3, ksize = [1, 2,2, 1], strides = [1, 2,2, 1], padding='SAME')

		#W4
		Z4 = tf.nn.conv2d(P3, W4, strides=[1, 2, 2, 1], padding='SAME')
		A4 = tf.nn.elu(Z4)
		P4 = tf.nn.max_pool(A4, ksize = [1, 4,4, 1], strides = [1, 2,2, 1], padding='SAME')

		#W5
		Z5 = tf.nn.conv2d(P4, W5, strides=[1,2, 2, 1], padding='SAME')
		A5 = tf.nn.elu(Z5)
		P5 = tf.nn.max_pool(A5, ksize = [1, 2,2, 1], strides = [1, 2,2, 1], padding='SAME')
		
		# FLATTEN
		P = tf.contrib.layers.flatten(P5)
		Z6 = tf.contrib.layers.fully_connected(P, 12, activation_fn=None)
		Z7 = tf.contrib.layers.fully_connected(Z6, 12, activation_fn=None)

	return Z7


# In[2]:


#saver = tf.train.Saver()
def model(X_train, Y_train, X_test, Y_test,learning_rate=0.009,
		  num_epochs=100, minibatch_size=64, print_cost=True, large_files = False, VAT = False):

	tf.reset_default_graph()
	print("Batch Size : {}\nEpochs: {}\nLearning Rate: {}\nVAT: {}\nLarge_Files: {} ".format(minibatch_size,num_epochs,learning_rate,VAT,large_files))
	title = "elu activations lr "+ str(learning_rate)+" mbs "+str(minibatch_size)+" e "+str(num_epochs)
	if large_files:
		title = "large images"+str(title)
	if VAT:
		title = "VAT "+str(title)
	ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
	tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
	seed = 3                                          # to keep results consistent (numpy seed)
	(m, n_H0, n_W0, n_C0) = X_train.shape             
	n_y = Y_train.shape[1]                            
	costs = []                                        # To keep track of the cost
	
	# Create Placeholders of the correct shape
	X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

	# Initialize parameters
	parameters = initialize_parameters()
	
	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z7 = forward_propagation(X, parameters)
	
	# Cost function: Add cost function to tensorflow graph
	#cost = compute_cost(Z3, Y)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z7, labels = Y))
	with tf.name_scope('Optimizer'):
	# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	#saver = tf.train.Saver()
	#summary_op = tf.summary.merge_all()
	config = tf.ConfigProto(allow_soft_placement = True,log_device_placement = False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	init = tf.global_variables_initializer()
	merged = tf.summary.merge([tf.summary.scalar('cross_entropy', cost)])
	#writer = tf.summary.FileWriter(os.path.join(os.getcwd(),"logs"), graph=sess.graph)
	with sess.as_default():
		#saver.restore(sess,str(title)+".ckpt")
		sess.run(init)
		# Do the training loop
		
		for epoch in range(num_epochs+1):
			start = time.time()
			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			batch_count = int(m/minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
			#c = 0
			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_ , temp_cost,summary= sess.run([optimizer, cost,merged], feed_dict={X:minibatch_X, Y:minibatch_Y})
				
				#c += 1
				minibatch_cost += temp_cost / num_minibatches
			#print(type(summary))
			end = time.time()
			if minibatch_cost <= 0.36:
				print("\n == end of training at epoch: {} with cost: {} ==".format(epoch,minibatch_cost))
				break;
			#writer.add_summary(summary, epoch)
			if print_cost == False:
				drawProgressBar(epoch/num_epochs,barLen = 50)
			# Print the cost every epoch
			if print_cost == True: 
				if num_epochs<100 and epoch % 2 == 0:
					print ("Cost after epoch {}: {}".format(epoch, minibatch_cost))
				elif num_epochs<1000:
					if epoch % 10 == 0:
					#end = time.time()
						print ("Cost after epoch {}: {}".format(epoch, minibatch_cost))
				elif num_epochs >= 1000:
					if epoch % 50 == 0:
						print("Cost after epoch {}:{}".format(epoch,minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)
		
		#with open(str(title)+".txt","w") as f:
			#for i in range(len(costs)):
				#f.write(str(costs[i]))
				#f.write("\n")
		# plot the cost
		plt.figure(figsize = (5,5))
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title(str(title))#"Learning rate =" + str(learning_rate))
		plt.savefig((title)+".png")
		plt.show()
		with tf.device('/device:GPU:0'):
			# Calculate the correct predictions
			predict_op = tf.argmax(Z7, 1)
			correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

			# Calculate accuracy on the test set
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print(accuracy)
			train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
			test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
		#saver.save(sess,str(title)+".ckpt")
		print("Saved Model at : {}.ckpt".format(str(title)))
		return train_accuracy, test_accuracy, parameters


# In[3]:


print("=====================")
print("Training without VAT")
print("=====================")
images,labels = load_train("im_train")
train_x,test_x,train_y,test_y = split_data(images,labels, test_size = 0.2, shuffle = True)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)

#print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))
start = time.time()
_,_, params_vat_large = model(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.001, num_epochs = 5000, minibatch_size = 1024, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60

images,labels = load_train("im_train_large")
train_x,test_x,train_y,test_y = split_data(images,labels, test_size = 0.2, shuffle = True)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
#print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))
start = time.time()
_,_, params_vat_large = model(train_x,train_y,test_x,test_y,VAT = False, large_files = True
                              ,learning_rate = 0.001, num_epochs = 5000, minibatch_size = 1024, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60
print("Total time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))


print("=====================")
print("Beginning VAT")
print("=====================")
images,labels = load_train("im_train")
train_x,test_x,train_y,test_y = split_data(images,labels, test_size = 0.2, shuffle = True)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
train_x2 = np.add(np.random.randn(train_x.shape[0],train_x.shape[1],train_x.shape[2],train_x.shape[3])*np.std(train_x)*0.01,train_x)
train_y2 = train_y
train_y3 = np.append(train_y,train_y2, axis = 0)
train_x3 = np.append(train_x,train_x2,axis = 0)
print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))
start = time.time()
_,_, params_vat_large = model(train_x3,train_y3,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.001, num_epochs = 5000, minibatch_size = 1024, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60
print("\n\n\nvat large files\n\n")
images,labels = load_train("im_train_large")
train_x,test_x,train_y,test_y = split_data(images,labels, test_size = 0.2, shuffle = True)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
train_x2 = np.add(np.random.randn(train_x.shape[0],train_x.shape[1],
                                  train_x.shape[2],train_x.shape[3])*np.std(train_x)*0.01,train_x)
train_y2 = train_y
train_y3 = np.append(train_y,train_y2, axis = 0)
train_x3 = np.append(train_x,train_x2,axis = 0)
print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))
start = time.time()
_,_, params_vat_large = model(train_x3,train_y3,test_x,test_y,VAT = True, large_files = True
                              ,learning_rate = 0.001, num_epochs = 5000, minibatch_size = 1024, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60
print("Total time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))


# In[4]:


plt.imshow(train_x[12])


# In[8]:


plt.imshow(train_x2[12])


# In[9]:


print(train_y[12],train_y2[12])


# In[11]:


plt.imshow(train_x[12]*np.std(train_x))


# In[16]:


plt.imshow(train_x[12]+(np.std(train_x)*0.0001))


# In[7]:


train_x[12] == train_x2[12]


# In[12]:


train_x2 = np.add(np.random.randn(train_x.shape[0],train_x.shape[1],train_x.shape[2],train_x.shape[3])*np.std(train_x)*0.01,train_x)
train_y2 = train_y
train_y3 = np.append(train_y,train_y2, axis = 0)
train_x3 = np.append(train_x,train_x2,axis = 0)

#print("testing on non VAT:")
#title = "temp"
print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))


_,_,_ = model(train_x,train_y,test_x,test_y, learning_rate = 0.001, num_epochs = 15 , minibatch_size = 64 , print_cost = True)

#print("lr009  batch_size 32 100 epochs")
#title = "lr 0009 e 100 mbs 2048"
#_,_,_ = model(train_x3,train_y3,test_x,test_y, learning_rate = 0.009,num_epochs = 100,minibatch_size = 64, print_cost = True)

#print("lr 0009 mbs e 1000 2048")

start = time.time()
#_, _, parameters = model(train_x3, train_y3, test_x, test_y,learning_rate=0.009,
#		  num_epochs = 1000, minibatch_size = 64, print_cost=True)
_,_, params = model(train_x3,train_y3,test_x,test_y,learning_rate = 0.001, num_epochs = 5000, minibatch_size = 1024, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60
print("Total time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))
#title = "lr 0001 mb 2048 ep 5000 adv training"
#print(title)


# In[13]:


images,labels = load_train("im_train_large")
train_x,test_x,train_y,test_y = split_data(images,labels, test_size = 0.2, shuffle = True)
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
train_x2 = np.add(np.random.randn(train_x.shape[0],train_x.shape[1],train_x.shape[2],train_x.shape[3])*np.std(train_x)*0.01,train_x)
train_y2 = train_y
train_y3 = np.append(train_y,train_y2, axis = 0)
train_x3 = np.append(train_x,train_x2,axis = 0)

#print("testing on non VAT:")
#title = "temp"
print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))


_,_,_ = model(train_x,train_y,test_x,test_y, learning_rate = 0.006, num_epochs = 20 , minibatch_size = 64 , print_cost = True)

#print("lr009  batch_size 32 100 epochs")
#title = "lr 0009 e 100 mbs 2048"
#_,_,_ = model(train_x3,train_y3,test_x,test_y, learning_rate = 0.009,num_epochs = 100,minibatch_size = 64, print_cost = True)

#print("lr 0009 mbs e 1000 2048")

start = time.time()
#_, _, parameters = model(train_x3, train_y3, test_x, test_y,learning_rate=0.009,
#		  num_epochs = 1000, minibatch_size = 64, print_cost=True)
_,_, params = model(train_x3,train_y3,test_x,test_y,learning_rate = 0.009, num_epochs = 3000, minibatch_size = 64, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60
print("Total time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))
#title = "lr 0001 mb 2048 ep 5000 adv training"
#print(title)


# In[15]:


#print("testing on non VAT:")
#title = "temp"
print("train_x3: {}\ntrain_y3: {}\ntest_x.shape: {}\ntest_y.shape: {}\n\n\n".format(train_x3.shape,train_y3.shape,test_x.shape,test_y.shape))


_,_,_ = model(train_x,train_y,test_x,test_y, learning_rate = 0.001, num_epochs = 15 , minibatch_size = 64 , print_cost = True)

#print("lr009  batch_size 32 100 epochs")
#title = "lr 0009 e 100 mbs 2048"
#_,_,_ = model(train_x3,train_y3,test_x,test_y, learning_rate = 0.009,num_epochs = 100,minibatch_size = 64, print_cost = True)

#print("lr 0009 mbs e 1000 2048")
print("\n\n\n\n\n")
start = time.time()
#_, _, parameters = model(train_x3, train_y3, test_x, test_y,learning_rate=0.009,
#		  num_epochs = 1000, minibatch_size = 64, print_cost=True)
_,_, params = model(train_x3,train_y3,test_x,test_y,learning_rate = 0.001, num_epochs = 5000, minibatch_size = 1024, print_cost = True)
total_end = time.time()
hrs = 0
mins = (total_end-start)/60
if mins > 60:
	hrs = mins/60
	mins %= 60
secs = (total_end-start)%60
print("Total time taken = %i hours, %i minutes and %.4f seconds"%(hrs,mins, secs))
#title = "lr 0001 mb 2048 ep 5000 adv training"
#print(title)

