# The office reference
# Script to recursively resize images and save them with a different name
# author : Sanjay Krishna 
# email : sgouda@ucsc.edu
import pandas as pd
import tensorflow as tf
import PIL,time,sys,os
from pathlib import Path
from PIL import Image

path = os.getcwd()
dirs = os.listdir( path )
#print(dirs)
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
del dirs[dirs.index('downsizing.py')]
del dirs[dirs.index('matrix.txt')]


subFolders = {}
for i in range(len(dirs)):
	if dirs[i] != 'downsizing.py' or dirs[i] != 'matrix.txt':
		subFolders[dirs[i]] = os.listdir(os.path.join(path,dirs[i]))
#for i in range(len(subFolders)):
#	print(subFolders[i])
#folders = list(subFolders.keys()) # same as dirs
for folder in dirs:
	filesList = os.listdir(os.path.join(path,folder))
	t = 0
	print("\nConverting files in ",folder)
	for file in filesList:
		drawProgressBar((t+1)/len(filesList))
		if os.path.isfile(os.path.join(path,folder,file)):
			im = Image.open(os.path.join(path,folder,file))
			f, e = os.path.splitext(os.path.join(path,folder,file))
			imResize = im.resize((28,28), Image.ANTIALIAS)
			imResize.save(os.path.join(path,folder,str(f+'_resized.png')), 'PNG', quality=90)
			t += 1
	print("\nDone")
c = 1
with open("matrix.txt", "w") as f:
	f.write("Name,Class,\n")
	print("Creating Matrix")
	for k,v in subFolders.items():
		drawProgressBar(c/len(subFolders.keys()))

		for i in range(len(v)):
			f.write("{},{}\n".format(v[i],k))
			#print(k,v[i])
		c += 1
	print("\nFinished Creating Matrix!")
#data = pd.DataFrame.from_dict(subFolders)

#print(data.head())