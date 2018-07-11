from __future__ import print_function, division

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image
from audiotospeech import drawProgressBar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tfSpeechDataSet(Dataset):
    """TF Speech recognition labels dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.to_tensor = transforms.ToTensor()
        self.imageFrame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_arr = np.asarray(self.imageFrame.iloc[:,1])
        self.label_arr = np.asarray(self.imageFrame.iloc[:,2])
        self.data_len = len(self.imageFrame.index)

    def __len__(self):
        return len(self.imageFrame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.image_arr[idx])
        image = Image.open(img_name).convert(mode='RGB')#io.imread(img_name)
        image_as_tensor = self.to_tensor(image)
        image_label = self.label_arr[idx]

        if self.transform:
            image_as_tensor = self.transform(image)

        return (image_as_tensor,image_label)
class utils():
    def __init__(self,dataset):
        self.dataset = dataset
        POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
        self.id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
        self.name2id = {name: i for i, name in self.id2name.items()}
    def show_wave(self):
        for i in range(1,len(self.dataset)):
            sample = self.dataset[i]
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(self.id2name[int(sample['labels'])]))
            ax.axis('off')
            # show_wave(sample['image'])
            if i == 3:
                plt.show()
                break
            plt.imshow(sample['image'])
            plt.pause(0.1)  # pause a bit so that plots are updated
    

def main():
    PATH = os.getcwd()+os.sep+'images'
    TRAINPATH = PATH+os.sep+'train'
    TESTPATH = PATH+os.sep+'test'
    VALIDATIONPATH = PATH+os.sep+'valid'

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])        
    }

    testdata = tfSpeechDataSet('test.csv',TESTPATH,transform=data_transforms['test'])
    traindata = tfSpeechDataSet('train.csv',TRAINPATH,transform=data_transforms['train'])
    validdata = tfSpeechDataSet('valid.csv',VALIDATIONPATH,transform=data_transforms['valid'])
    # image_datasets = {'test': testdata,
    #                     'train': traindata,
    #                     'valid': validdata}

    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
    #                                          shuffle=True, num_workers=4)
    #           for x in ['train', 'valid']}                 


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid','test']:
                if phase == 'train':
                    dataloaders = torch.utils.data.DataLoader(dataset=traindata,
                                                                    batch_size=32,
                                                                    shuffle=False)
                    scheduler.step()
                    model.train()  # Set model to training mode
                elif phase == 'valid':
                    dataloaders = torch.utils.data.DataLoader(dataset=validdata,
                                                                    batch_size=32,
                                                                    shuffle=False)
                    model.eval()   # Set model to evaluate mode
                else:
                    dataloaders = torch.utils.data.DataLoader(dataset=testdata,
                                                                    batch_size=32,
                                                                    shuffle=False)
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i,(inputs, labels) in enumerate(dataloaders):
                    drawProgressBar(i/len(dataloaders))
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders)
                epoch_acc = running_corrects.double() / len(dataloaders)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

if __name__ == '__main__':
    main()