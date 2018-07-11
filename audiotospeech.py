from preprocess import *
import pickle
import os
import gc
import time

# def log_specgram(audio,sample_rate = 8, window_size = 25, step_size = 10, eps = 1e-10):
# def wav2img(wav_path, targetdir='./test_pics',figsize=(2,2))
def drawProgressBar(percent, barLen = 50,key = None):
	sys.stdout.write("\r")
	progress = ""
	for i in range(barLen):
		if i<int(barLen * percent):
			progress += "="
		else:
			progress += " "
	if key:
		sys.stdout.write("%s [ %s ] %.2f%%" % (key,progress, percent * 100))
	else:
		sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
	sys.stdout.flush()
def calls(dictionary = None, TYPE = None):
	
	TRAINPATH = os.getcwd()+os.sep+'train'+os.sep+'audio'
	if TYPE in ['train','valid','test']:
		SAVEPATH = os.getcwd()+os.sep+'images'+os.sep+str(TYPE)
		for key in dictionary.keys():
			files = dictionary[key]
			for i,f in enumerate(files):
				drawProgressBar(i/len(files),key = key)
				path = TRAINPATH+os.sep+str(key)+os.sep+f
				Preprocess.wav2img(path,targetdir=SAVEPATH)
	else:
		raise ValueError("train/valid/test type")
	return


def main():
	TRAINPATH = os.getcwd()+os.sep+'train'+os.sep+'audio'
	train = pickle.load(open('train.dict','rb'))
	valid = pickle.load(open('validation.dict','rb'))
	test = pickle.load(open('test.dict','rb'))
	
	calls(dictionary=test,TYPE='test')
	print("\nFinished Converting test data")
	del test
	gc.collect()

	calls(dictionary=valid,TYPE='valid')
	del valid
	gc.collect()
	print("\nFinished Converting valid data")
	
	calls(dictionary=train,TYPE='train')
	print("\nFinished Converting train data")
	del train
	gc.collect()

	
	

if __name__ == "__main__":
	start = time.time()
	main()
	end = time.time()

	seconds = end - start
	hours = 0
	minutes,seconds = divmod(seconds,60)
	if minutes >=60:
		hours , minutes = divmod(minutes,60)
	print("\n")
	print("______________________")
	print("Time taken: {}:{}:{}".format(hours,minutes,seconds))
	print("______________________")