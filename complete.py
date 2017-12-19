#Author: Sanjay Krishna Gouda
# sgouda@ucsc.edu
# Will structure the code if I get time
# coding: utf-8


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
import matplotlib.pyplot as plt
import seaborn as sns
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


# In[7]:


images, labels = load_train("im_train") #loading the image data
train_x, test_x, train_y, test_y = split_data(images, labels,test_size = 0.2
                                              , random_state = 7, shuffle = True) #creating training and validation sets
train_x_temp = np.concatenate((train_x,np.add(train_x, train_x*0.02*np.std(train_x))),axis = 0) # creating VAs of training set and concatanating
train_y_temp = train_y #VAs to have same labels
train_y_temp = np.concatenate((train_y_temp,train_y_temp),axis = 0)
random_indices = np.random.choice(np.arange(len(train_x)), len(train_x), replace = False)
train_x_vat, train_y_vat = train_x_temp[random_indices], train_y_temp[random_indices]

""" 
################ deprecated ################

# random sampling from appended matrix to get equal size training data for vanilla and VAT
#checking if above method worked.. 
train_check = [train_x_vat[i] == train_x[random_indices[i]] for i in range(len(random_indices))]
print("train sampling passed? : ",np.all(train_check))
labels_check = [train_y_vat[i] == train_y[random_indices[i]] for i in range(len(random_indices))]
print("label sampling passed?:",np.all(labels_check))

################ deprecated ################
"""


def initialize_parameters_1(): #parameters for a deeper, wider CNN
    tf.set_random_seed(1)
    with tf.device('/device:CPU:0'):
        W1 = tf.get_variable("W1",[7, 7, 3, 64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W2 = tf.get_variable("W2",[5, 5, 64, 32], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W3 = tf.get_variable("W3",[3, 3, 32, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W4 = tf.get_variable("W4",[2, 2, 16, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W5 = tf.get_variable("W5",[2, 2, 8, 4], initializer = tf.contrib.layers.xavier_initializer(seed = 0))


    parameters = {"W1": W1,
                  "W2": W2,
                  "W3":W3,
                  "W4":W4,
                  "W5":W5}

    return parameters


# In[13]:


def forward_propagation_1(X, parameters):

	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']
	W5 = parameters['W5']
	with tf.device('/device:CPU:0'): #GPU to CPU

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


# In[10]:


def model_1(X_train, Y_train, X_test, Y_test,learning_rate=0.009,
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
	parameters = initialize_parameters_1()
	
	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z7 = forward_propagation_1(X, parameters)
	
	# Cost function: Add cost function to tensorflow graph
	#cost = compute_cost(Z3, Y)
	with tf.device('/device:GPU:0'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z7, labels = Y))
	with tf.name_scope('Optimizer') and tf.device('/device:GPU:0'):
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
	training_accs = []
	validation_accs = []
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
				_ , temp_cost= sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
				
				#c += 1
				minibatch_cost += temp_cost / num_minibatches
			#print(type(summary))
			end = time.time()
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
					if epoch % 100 == 0:
						print("Cost after epoch {}:{}".format(epoch,minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)


			with tf.device('/device:GPU:0'): # GPU to CPU
				# Calculate the correct predictions
				predict_op = tf.argmax(Z7, 1)
				correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
				# Calculate accuracy on the test set
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				#print(accuracy)
				train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
				test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
				training_accs.append(train_accuracy)
				validation_accs.append(test_accuracy)
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
		plt.figure(figsize = (5,5))
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("larger CNN run 3\n "+str(title)+"\n training acc: {:.4f} validation acc: {:.4f}".format(train_accuracy,test_accuracy))#"Learning rate =" + str(learning_rate))
		plt.savefig("larger CNN run 3 "+(title)+".png")
		plt.figure(figsize = (6,6))
		plt.plot(training_accs, label = "Training acc: {0:3f}".format(training_accs[-1]))
		plt.plot(validation_accs, label = "Validation acc: {0:3f}".format(validation_accs[-1]))
		plt.legend()
		#plt.savefig("larger CNN run 3 tr v "+(title)+".png")        
		return train_accuracy, test_accuracy, parameters, training_accs, validation_accs, costs


# In[16]:


def initialize_parameters_2():
    tf.set_random_seed(1)
    with tf.device('/device:CPU:0'):
        W1 = tf.get_variable("W1",[7, 7, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W2 = tf.get_variable("W2",[5, 5, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W3 = tf.get_variable("W3",[3, 3, 16, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3":W3}

    return parameters
def forward_propagation_2(X, parameters):

	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']

	with tf.device('/device:GPU:0'): #GPU to CPU -> GPU. Cause, small

		Z1 = tf.nn.conv2d(X,W1,strides = [1,2,2,1], padding = 'SAME')
		A1 = tf.nn.elu(Z1)		
		P1 = tf.nn.max_pool(A1,ksize = [1, 8, 8,1], strides = [1,8,8,1],padding = 'SAME')
		

		Z2 = tf.nn.conv2d(P1, W2, strides=[1,2,2, 1], padding='SAME')
		A2 = tf.nn.elu(Z2)#relu(Z2)
		P2 = tf.nn.max_pool(A2, ksize = [1, 4,4,1], strides = [1, 2,2, 1], padding='SAME')
		
		Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
		A3 = tf.nn.elu(Z3)
		P3 = tf.nn.max_pool(A3, ksize = [1, 2,2, 1], strides = [1, 2,2, 1], padding='SAME')

		# FLATTEN
		P = tf.contrib.layers.flatten(P3)
		Z4 = tf.contrib.layers.fully_connected(P, 12, activation_fn=None)
		Z5 = tf.contrib.layers.fully_connected(Z4, 12, activation_fn=None)

	return Z5


# In[20]:


def model_2(X_train, Y_train, X_test, Y_test,learning_rate=0.009,
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
	parameters = initialize_parameters_2()

	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z5 = forward_propagation_2(X, parameters)
	
	# Cost function: Add cost function to tensorflow graph
	#cost = compute_cost(Z3, Y)
	with tf.device('/device:GPU:0'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y))
	with tf.name_scope('Optimizer') and tf.device('/device:GPU:0'):
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
	training_accs = []
	validation_accs = []
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
				_ , temp_cost= sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
				
				#c += 1
				minibatch_cost += temp_cost / num_minibatches
			#print(type(summary))
			end = time.time()
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
					if epoch % 100 == 0:
						print("Cost after epoch {}:{}".format(epoch,minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)


			with tf.device('/device:GPU:0'): # GPU to CPU
				# Calculate the correct predictions
				predict_op = tf.argmax(Z5, 1)
				correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
				# Calculate accuracy on the test set
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				#print(accuracy)
				train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
				test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
				training_accs.append(train_accuracy)
				validation_accs.append(test_accuracy)
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
		plt.figure(figsize = (5,5))
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("larger CNN run 3\n "+str(title)+"\n training acc: {:.4f} validation acc: {:.4f}".format(train_accuracy,test_accuracy))#"Learning rate =" + str(learning_rate))
		plt.savefig("larger CNN run 3 "+(title)+".png")
		plt.figure(figsize = (6,6))
		plt.plot(training_accs, label = "Training acc: {0:3f}".format(training_accs[-1]))
		plt.plot(validation_accs, label = "Validation acc: {0:3f}".format(validation_accs[-1]))
		plt.legend()
		#plt.savefig("larger CNN run 3 tr v "+(title)+".png")        
		return train_accuracy, test_accuracy, parameters, training_accs, validation_accs, costs


# In[38]:


#only smaller images
#both vanilla and VAT
#both small and large architectures -> use model 1 for large, model 2 for small
#tal -> training accs list, val -> validation accs list

#vanilla CNN small
training_accuracy_vanilla_small, testing_accuracy_vanilla_small,parameters_vanilla_small, tal_vanilla_small,val_vanilla_small,costs_vanilla_small = model_2(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.009, num_epochs = 10, minibatch_size = 1024, print_cost = True)
#VAT for same CNN
training_accuracy_vat_small, testing_accuracy_vat_small,parameters_vat_small, tal_vat_small,val_vat_small,costs_vat_small = model_2(train_x_vat,train_y_vat,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.009, num_epochs = 10, minibatch_size = 1024, print_cost = True)
#vanilla large CNN
training_accuracy_vanilla_large, testing_accuracy_vanilla_large,parameters_vanilla_large, tal_vanilla_large,val_vanilla_large,costs_vanilla_large = model_2(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.009, num_epochs = 10, minibatch_size = 1024, print_cost = True)
#VAT for same CNN
training_accuracy_vat_large, testing_accuracy_vat_large,parameters_vat_large, tal_vat_large,val_vat_large,costs_vat_large = model_2(train_x_vat,train_y_vat,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.009, num_epochs = 10, minibatch_size = 1024, print_cost = True)
plt.figure(figsize=(10,10))
plt.title("Smaller CNN training vs validation ")
plt.plot(tal_vanilla_small)
plt.plot(val_vanilla_small)
plt.plot(tal_vat_small)
plt.plot(val_vat_small)
plt.legend(['Vanilla Training = {0:.3f}'.format(training_accuracy_vanilla_small),
            'Vanilla Validation = {0:.3f}'.format(testing_accuracy_vanilla_small),
            'VAT Training = {0:.3f}'.format(training_accuracy_vat_small)
            ,'VAT Validation = {0:.3f}'.format(testing_accuracy_vat_small)])
plt.figure(figsize=(10,10))
plt.title("Larger CNN training vs validation ")
plt.plot(tal_vanilla_large)
plt.plot(val_vanilla_large)
plt.plot(tal_vat_large)
plt.plot(val_vat_large)
plt.legend(['Vanilla Training * = {0:.3f}'.format(training_accuracy_vanilla_large),
            'Vanilla Validation * = {0:.3f}'.format(testing_accuracy_vanilla_large),
            'VAT Training * = {0:.3f}'.format(training_accuracy_vat_large)
            ,'VAT Validation * = {0:.3f}'.format(testing_accuracy_vat_large)])


# In[39]:


plt.figure(figsize=(10,10))
plt.title("training vs validation 10 epochs")
plt.plot(tal_vanilla_small)
plt.plot(val_vanilla_small)
plt.plot(tal_vat_small)
plt.plot(val_vat_small)
plt.plot(tal_vanilla_large)
plt.plot(val_vanilla_large)
plt.plot(tal_vat_large)
plt.plot(val_vat_large)
plt.legend(['Vanilla Training = {0:.3f}'.format(training_accuracy_vanilla_small),
            'Vanilla Validation = {0:.3f}'.format(testing_accuracy_vanilla_small),
            'VAT Training = {0:.3f}'.format(training_accuracy_vat_small)
            ,'VAT Validation = {0:.3f}'.format(testing_accuracy_vat_small),'Vanilla Training * = {0:.3f}'.format(training_accuracy_vanilla_large),
            'Vanilla Validation * = {0:.3f}'.format(testing_accuracy_vanilla_large),
            'VAT Training * = {0:.3f}'.format(training_accuracy_vat_large)
            ,'VAT Validation * = {0:.3f}'.format(testing_accuracy_vat_large)])


# In[41]:


plt.figure(figsize = (6,6))
plt.title("costs 10 epochs")
plt.plot(costs_vanilla_small)
plt.plot(costs_vanilla_large)
plt.plot(costs_vat_small)
plt.plot(costs_vat_large)
plt.legend(['costs_vanilla_small'.format(costs_vanilla_small[-1]),
            'costs_vanilla_large'.format(costs_vanilla_large)[-1],
            'costs_vat_small'.format(costs_vat_small[-1]),
            'costs_vat_large'.format(costs_vat_large[-1])])


# In[42]:


#only smaller images
#both vanilla and VAT
#both small and large architectures -> use model 1 for large, model 2 for small
#tal -> training accs list, val -> validation accs list

#vanilla CNN small
training_accuracy_vanilla_small, testing_accuracy_vanilla_small,parameters_vanilla_small, tal_vanilla_small,val_vanilla_small,costs_vanilla_small = model_2(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.009, num_epochs = 200, minibatch_size = 1024, print_cost = True)
#VAT for same CNN
training_accuracy_vat_small, testing_accuracy_vat_small,parameters_vat_small, tal_vat_small,val_vat_small,costs_vat_small = model_2(train_x_vat,train_y_vat,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.009, num_epochs = 200, minibatch_size = 1024, print_cost = True)
#vanilla large CNN
training_accuracy_vanilla_large, testing_accuracy_vanilla_large,parameters_vanilla_large, tal_vanilla_large,val_vanilla_large,costs_vanilla_large = model_2(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.009, num_epochs = 200, minibatch_size = 1024, print_cost = True)
#VAT for same CNN
training_accuracy_vat_large, testing_accuracy_vat_large,parameters_vat_large, tal_vat_large,val_vat_large,costs_vat_large = model_2(train_x_vat,train_y_vat,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.009, num_epochs = 200, minibatch_size = 1024, print_cost = True)
plt.figure(figsize=(10,10))
plt.title("Smaller CNN training vs validation 200 epochs ")
plt.plot(tal_vanilla_small)
plt.plot(val_vanilla_small)
plt.plot(tal_vat_small)
plt.plot(val_vat_small)
plt.legend(['Vanilla Training = {0:.3f}'.format(training_accuracy_vanilla_small),
            'Vanilla Validation = {0:.3f}'.format(testing_accuracy_vanilla_small),
            'VAT Training = {0:.3f}'.format(training_accuracy_vat_small)
            ,'VAT Validation = {0:.3f}'.format(testing_accuracy_vat_small)])
plt.figure(figsize=(10,10))
plt.title("Larger CNN training vs validation 200 epochs")
plt.plot(tal_vanilla_large)
plt.plot(val_vanilla_large)
plt.plot(tal_vat_large)
plt.plot(val_vat_large)
plt.legend(['Vanilla Training * = {0:.3f}'.format(training_accuracy_vanilla_large),
            'Vanilla Validation * = {0:.3f}'.format(testing_accuracy_vanilla_large),
            'VAT Training * = {0:.3f}'.format(training_accuracy_vat_large)
            ,'VAT Validation * = {0:.3f}'.format(testing_accuracy_vat_large)])
plt.figure(figsize=(15,15))
plt.title("training vs validation 200 epochs")
plt.plot(tal_vanilla_small)
plt.plot(val_vanilla_small)
plt.plot(tal_vat_small)
plt.plot(val_vat_small)
plt.plot(tal_vanilla_large)
plt.plot(val_vanilla_large)
plt.plot(tal_vat_large)
plt.plot(val_vat_large)
plt.legend(['Vanilla Training = {0:.3f}'.format(training_accuracy_vanilla_small),
            'Vanilla Validation = {0:.3f}'.format(testing_accuracy_vanilla_small),
            'VAT Training = {0:.3f}'.format(training_accuracy_vat_small)
            ,'VAT Validation = {0:.3f}'.format(testing_accuracy_vat_small),'Vanilla Training * = {0:.3f}'.format(training_accuracy_vanilla_large),
            'Vanilla Validation * = {0:.3f}'.format(testing_accuracy_vanilla_large),
            'VAT Training * = {0:.3f}'.format(training_accuracy_vat_large)
            ,'VAT Validation * = {0:.3f}'.format(testing_accuracy_vat_large)])
plt.figure(figsize = (8,8))
plt.title("costs 200 epochs")
plt.plot(costs_vanilla_small)
plt.plot(costs_vanilla_large)
plt.plot(costs_vat_small)
plt.plot(costs_vat_large)
plt.legend(['costs_vanilla_small'.format(costs_vanilla_small[-1]),
            'costs_vanilla_large'.format(costs_vanilla_large)[-1],
            'costs_vat_small'.format(costs_vat_small[-1]),
            'costs_vat_large'.format(costs_vat_large[-1])])


# In[43]:


#only smaller images
#both vanilla and VAT
#both small and large architectures -> use model 1 for large, model 2 for small
#tal -> training accs list, val -> validation accs list

#vanilla CNN small
training_accuracy_vanilla_small, testing_accuracy_vanilla_small,parameters_vanilla_small, tal_vanilla_small,val_vanilla_small,costs_vanilla_small = model_2(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.009, num_epochs = 500, minibatch_size = 1024, print_cost = True)
#VAT for same CNN
training_accuracy_vat_small, testing_accuracy_vat_small,parameters_vat_small, tal_vat_small,val_vat_small,costs_vat_small = model_2(train_x_vat,train_y_vat,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.009, num_epochs = 500, minibatch_size = 1024, print_cost = True)
#vanilla large CNN
training_accuracy_vanilla_large, testing_accuracy_vanilla_large,parameters_vanilla_large, tal_vanilla_large,val_vanilla_large,costs_vanilla_large = model_2(train_x,train_y,test_x,test_y,VAT = False, large_files = False
                              ,learning_rate = 0.009, num_epochs = 500, minibatch_size = 1024, print_cost = True)
#VAT for same CNN
training_accuracy_vat_large, testing_accuracy_vat_large,parameters_vat_large, tal_vat_large,val_vat_large,costs_vat_large = model_2(train_x_vat,train_y_vat,test_x,test_y,VAT = True, large_files = False
                              ,learning_rate = 0.009, num_epochs = 500, minibatch_size = 1024, print_cost = True)
plt.figure(figsize=(10,10))
plt.title("Smaller CNN training vs validation 200 epochs ")
plt.plot(tal_vanilla_small)
plt.plot(val_vanilla_small)
plt.plot(tal_vat_small)
plt.plot(val_vat_small)
plt.legend(['Vanilla Training = {0:.3f}'.format(training_accuracy_vanilla_small),
            'Vanilla Validation = {0:.3f}'.format(testing_accuracy_vanilla_small),
            'VAT Training = {0:.3f}'.format(training_accuracy_vat_small)
            ,'VAT Validation = {0:.3f}'.format(testing_accuracy_vat_small)])
plt.figure(figsize=(10,10))
plt.title("Larger CNN training vs validation 500 epochs")
plt.plot(tal_vanilla_large)
plt.plot(val_vanilla_large)
plt.plot(tal_vat_large)
plt.plot(val_vat_large)
plt.legend(['Vanilla Training * = {0:.3f}'.format(training_accuracy_vanilla_large),
            'Vanilla Validation * = {0:.3f}'.format(testing_accuracy_vanilla_large),
            'VAT Training * = {0:.3f}'.format(training_accuracy_vat_large)
            ,'VAT Validation * = {0:.3f}'.format(testing_accuracy_vat_large)])
plt.figure(figsize=(10,10))
plt.title("training vs validation 500 epochs")
plt.plot(tal_vanilla_small)
plt.plot(val_vanilla_small)
plt.plot(tal_vat_small)
plt.plot(val_vat_small)
plt.plot(tal_vanilla_large)
plt.plot(val_vanilla_large)
plt.plot(tal_vat_large)
plt.plot(val_vat_large)
plt.legend(['Vanilla Training = {0:.3f}'.format(training_accuracy_vanilla_small),
            'Vanilla Validation = {0:.3f}'.format(testing_accuracy_vanilla_small),
            'VAT Training = {0:.3f}'.format(training_accuracy_vat_small)
            ,'VAT Validation = {0:.3f}'.format(testing_accuracy_vat_small),'Vanilla Training * = {0:.3f}'.format(training_accuracy_vanilla_large),
            'Vanilla Validation * = {0:.3f}'.format(testing_accuracy_vanilla_large),
            'VAT Training * = {0:.3f}'.format(training_accuracy_vat_large)
            ,'VAT Validation * = {0:.3f}'.format(testing_accuracy_vat_large)])
plt.figure(figsize = (8,8))
plt.title("costs 500 epochs")
plt.plot(costs_vanilla_small)
plt.plot(costs_vanilla_large)
plt.plot(costs_vat_small)
plt.plot(costs_vat_large)
plt.legend(['costs_vanilla_small'.format(costs_vanilla_small[-1]),
            'costs_vanilla_large'.format(costs_vanilla_large)[-1],
            'costs_vat_small'.format(costs_vat_small[-1]),
            'costs_vat_large'.format(costs_vat_large[-1])])

