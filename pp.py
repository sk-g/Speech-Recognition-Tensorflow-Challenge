#some preprocessing on the produced log spectrum images
# author : Sanjay Krishna 
# email : sgouda@ucsc.edu
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import *
import sys,os,time,itertools,operator,glob,random
if os.name != 'posix':
	os.chdir(r"M:\Course stuff\Fall 17\CMPS 242\final project\picts288x288\train")
path = os.getcwd()
foldersList = os.listdir( path )
del foldersList[foldersList.index('downsizing.py')]
del foldersList[foldersList.index('matrix.txt')]
del foldersList[foldersList.index('pp.py')]

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
print("subFolders Array")
for i in range(len(foldersList)):
	print(foldersList[i])

#renaming files
for folder in foldersList:
	#c = 0
	curr_folder_path = os.path.join(path,folder)
	filesList = os.listdir(curr_folder_path)
	for file in filesList:
		f = os.path.join(curr_folder_path,file)
		curr_name = f.split(os.sep)[-1]
		new_name = str(folder)+"_"+str(curr_name)
		new_filePath = f.replace(curr_name,new_name)
		os.rename(f,new_filePath)
if os.name != 'posix':
	images_path = r"M:\Course stuff\Fall 17\CMPS 242\final project\im_train"
else:
	images_path = r"\media\sanjay"
os.chdir(images_path)
print(len(os.listdir(os.getcwd()))) # should be same as number of training examples

# create an empty dataframe with index file name and label columns 
df = pd.DataFrame(columns=('index','fname','label'))
# add corresponding filename parts as an entry in dataframe
curr_path = os.getcwd()
files = os.listdir(curr_path)
print("Building the dataframe\n")
start = time.time()
for i in range(len(files)):
    drawProgressBar(i/len(files),barLen = 80)
    df.loc[-1] = [i,str(files[i]),files[i].split('_')[0]]
    df.index = df.index + 1
    #df['label'] = files[i].split('_')[0]
end = time.time()
print("\nBuilding DataFrame done!\nTime taken = %d seconds"%(end-start))