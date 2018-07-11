import os
import sys
import collections
import pickle

import numpy as np

from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
from PIL import Image
from matplotlib.backend_bases import RendererBase
import matplotlib.pyplot as plt

class Preprocess():
    def __init__(self,dir):
        self.PATHS = [os.getcwd()+os.sep+i for i in ['train','test']]
        self.validationFiles = [i[i.index('/')+1:].strip() for i in open('validation_list.txt','r').readlines()]
        self.testFiles = [i[i.index('/')+1:].strip() for i in open('testing_list.txt','r').readlines()]
        self.trainFiles = None
    
    def getFiles(self):
        ## directory structure:
        #  train\audio\folders
        #  folder name = class
        #  folder\fname.wav = audio file

        # return dictionary : {class:[files]}

        trainDict = collections.defaultdict(list)
        validDict = collections.defaultdict(list)
        testDict = collections.defaultdict(list)
        tarin_path = os.getcwd()+os.sep+'train'+os.sep+'audio'
        folders = os.listdir(tarin_path)
        
        for folder in folders:
            if folder != '_background_noise_':
                path = tarin_path+os.sep+folder
                for f in os.listdir(path):
                    if f in self.validationFiles:
                        validDict[folder] += f,
                    elif f in self.testFiles:
                        testDict[folder] += f,                        
                    else:
                        trainDict[folder] += f,
        return trainDict,validDict,testDict


    @staticmethod
    def log_specgram(audio,sample_rate = 8, window_size = 25, step_size = 10, eps = 1e-10):
        """
        int FS=8;                    // default sampling rate in KHz, actual value will be obtained from wave file
        int HIGH=4;                   // default high frequency limit in KHz
        int LOW=0;                    // default low frequency limit in KHz
        int FrmLen=25;             // frame length in ms
        int FrmSpace=10;           // frame space in ms
        const unsigned long FFTLen=512;           // FFT points
        const double PI=3.1415926536;
        const int FiltNum=26;              // number of filters
        const int PCEP=12;                 // number of cepstrum
        vector <double> Hamming;            // Hamming window vector
        """
        nperseg = int(round(window_size * sample_rate /1e3))
        noverlap = int(round(step_size * sample_rate /1e3))
        freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window = 'hann',
                                        noverlap = noverlap,
                                        nperseg = nperseg,
                                        detrend = False)
        return freqs, np.log(spec.T.astype(np.float32)+eps)
    
    @staticmethod
    def wav2img(wav_path, 
        targetdir='./test_pics',
        figsize=(2,2)):
        
        fig = plt.figure(figsize = figsize)
        samplerate, test_sound = wavfile.read(wav_path)
        _, spectrogram = Preprocess.log_specgram(test_sound,samplerate)
        #o_f = os.path.join(targetdir,wav_path.split(os.sep)[-1].split('.wav')[0])
        class_key = wav_path.split(os.sep)[-2]
        o_f = wav_path.split(os.sep)[-1].split('.')[0]
        plt.imshow(spectrogram.T, aspect = 'equal', origin = 'lower')
        plt.imsave(targetdir+os.sep+class_key+'_'+'%s.png'%o_f,spectrogram)
        plt.close()

def print_dicts(kv):
    print([(k,len(v)) for k,v in kv.items()]) # test, OK

def main():
    obj = Preprocess(os.getcwd())

    # check validation files and test files
    # test ok
    print("\nValidation files: {}".format(len(obj.validationFiles)))
    print("\nTest files: {}".format(len(obj.testFiles)))
    
    ## get training files and corresponding class
    # test ok
    # Validation files: 6798
    # Test files: 6835
    # [('bed', 1713), ('bird', 1731), ('cat', 1733), ('dog', 1746), 
    #       ('down', 2359), ('eight', 2352), ('five', 2357), ('four', 2372), 
    #       ('go', 2372), ('happy', 1742), ('house', 1750), ('left', 2353), 
    #       ('marvin', 1746), ('nine', 2364), ('no', 2375), ('off', 2357), ('on', 2367), 
    #       ('one', 2370), ('right', 2367), ('seven', 2377), ('sheila', 1734), ('six', 2369), ('stop', 2380), 
    #       ('three', 2356), ('tree', 1733), ('two', 2373), ('up', 2375), ('wow', 1745), ('yes', 2377), ('zero', 2376)]    
    train,valid,test = obj.getFiles()
    pickle.dump(train, file = open('train.dict','wb'))
    pickle.dump(valid, file = open('validation.dict','wb'))
    pickle.dump(test, file = open('test.dict','wb'))
    print("\nTraining files:")
    print_dicts(train)
    print("\nvalidation files:")
    print_dicts(valid)
    print("\nTest files:")
    print_dicts(test)
if __name__ == '__main__':
    main()