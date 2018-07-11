import os
import glob

import numpy as np
import pandas as pd

## some code from https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72



## create csv file that looks like
# imageName     class
# bed_xxx.png   bed

# better yet
# imageName     class
# bed_xxx.png   1

def image2csv(what = None):
    if not what or what not in ['train','test','valid']:
        raise ValueError("Expected train/test/valid got None")
    
    else:
        fname = str(what)+'.csv'

    POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
    id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
    name2id = {name: i for i, name in id2name.items()}
    
    PATH = os.getcwd()+os.sep+'images'
    TRAINPATH = PATH+os.sep+'train'
    TESTPATH = PATH+os.sep+'test'
    VALIDATIONPATH = PATH+os.sep+'valid'
    if what == 'train':
        USEPATH = TRAINPATH
    elif what == 'test':
        USEPATH = TESTPATH
    elif what == 'valid':
        USEPATH = VALIDATIONPATH
    else:
        raise ValueError("Expected train/test/valid got None")
    
    files = os.listdir(USEPATH)
    classes = []
    images = []
    for f in files:
        if 'back' in f:
            getClass = 'silence'
        else:
            split = f.split('_')
            getClass = split[0]
        if getClass not in POSSIBLE_LABELS:
            getClass = 'unknown'
        else:
            getClass = getClass
        
        classes.append(getClass)
        images.append(f)
    for i in range(len(classes)):
        classes[i] = name2id[classes[i]]
        #print(images[i],classes[i])
    # df = pd.DataFrame(
    # {
    #     'image':images,
    #     'label': classes[i]
    # }
    # )
    df = pd.DataFrame(list(zip(images,classes)),
              columns=['image','label'])
    df.to_csv(fname)
    # print(df.head(10))
if __name__ == '__main__':
    # image2csv(what = 'test')
    # image2csv(what = 'valid')
    image2csv(what = 'train')