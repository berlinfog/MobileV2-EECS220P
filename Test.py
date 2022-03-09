#Importing Libraries
# from threading import main_thread

# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns
# from collections import Counter
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms,models
from torch.utils.data import DataLoader
import torch.optim as optim
import xmltodict as xt
from PIL import Image
import numpy as np
from ResNetshortcut import *
from ResNetresize import *
from MobileNet import *
from Standard import *
from ResNetdel import *
from Alexnetself import *
import torchsummary
from MobileNetV2 import *
import gc
# from torchvision.models import alexnet
#print("PyTorch Version: ",torch.__version__)
#!pip install xmltodict
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()
# Hyperparameters
random_seed = 123
# learning_rate = 0.01
learning_rate = 0.01
num_epochs = 1
batch_size = 64

# Architecture
num_classes = 3

def preprocess(): 
    path_image='../face_mask_detection/images/'
    path_xml='../face_mask_detection/annotations/'
    img_names=[] 
    for _, _, filenames in os.walk(path_image):
        for i in filenames:
            img_names.append(i)
    # append all mask or not information 
    options={"with_mask":0,"without_mask":1,"mask_weared_incorrect":2} #
    image_tensor=[]
    mask_label_tensor=[]
    my_transform=transforms.Compose([transforms.Resize((228,228)),transforms.ToTensor()])#convert to 226*226 input size
    for i,j in enumerate(img_names):
        with open(path_xml+j[:-4]+".xml") as fd:
            doc=xt.parse(fd.read())
        mask_info=doc["annotation"]["object"]
        
        if type(mask_info)!=type([]):#only one mask
            label=options[mask_info["name"]]
            xmin,ymin,xmax,ymax=list(map(int,mask_info["bndbox"].values()))    
            image=transforms.functional.crop(Image.open(path_image+j).convert("RGB"), ymin,xmin,ymax-ymin,xmax-xmin)#convert the mask cut image to rgb image 
            image_tensor.append(my_transform(image))#save the image information
            mask_label_tensor.append(torch.tensor(label))
        else:
            for k in range(len(mask_info)):#many mask
                label=options[mask_info[k]["name"]]
                xmin,ymin,xmax,ymax=list(map(int,mask_info[k]["bndbox"].values()))
                image=transforms.functional.crop(Image.open(path_image+j).convert("RGB"),ymin,xmin,ymax-ymin,xmax-xmin)
                image_tensor.append(my_transform(image))
                mask_label_tensor.append(torch.tensor(label))    
    final_dataset=[[k,l] for k,l in zip(image_tensor,mask_label_tensor)]
    return tuple(final_dataset)

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):            
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


if __name__ == '__main__':
    #preprocess
    mydataset=preprocess()
    # print(mydataset[0])

    #divide into train and test
    train_size=int(len(mydataset)*0.8)#one image may have several masks, so the size is bigger than 821
    test_size=len(mydataset)-train_size
    print('All dataset is', len(mydataset), ' pics \n Training set is :',train_size,' pics \nTest set is :', test_size, " pics")

    traindataset,testdataset=torch.utils.data.random_split(mydataset,[train_size,test_size])
    train_dataloader =DataLoader(dataset=traindataset,batch_size=32,shuffle=True,num_workers=4)#11800 has 8 core,set num workers as 8
    test_dataloader =DataLoader(dataset=testdataset,batch_size=32,shuffle=True,num_workers=4)


    for images, labels in train_dataloader:  
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    # start train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda
    torch.manual_seed(random_seed)
    # model = ConvNetre(num_classes=num_classes)
    # model = MobileNetV1Hand(ch_in=3, n_classes=num_classes)
    # model =AlexNetself()
    # model=StandardConv(ch_in=3, n_classes=num_classes)
    model=ConvNetre(num_classes=3)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    # model(num_classes=3)
    # torch.cuda.empty_cache()
    # model =MobileNetV2()
    # model=ConvNetre(num_classes=3)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
    # optimizer=optim.Adamax(model.parameters(),weight_decay=0.0003)
    # optimizer=optim.ASGD(model.parameters())
    start_time = time.time()
    for epoch in range(num_epochs):
        model = model.train()
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            
            features = features.to(device)
            targets = targets.to(device)
            
            ### FORWARD AND BACK PROP
            logits = model(features)
            cost = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 250:
                print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                       %(epoch+1, num_epochs, batch_idx, 
                         len(train_dataloader), cost))

        model = model.eval() # eval mode to prevent upd. batchnorm params during inference
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
                  epoch+1, num_epochs, 
                  compute_accuracy(model, test_dataloader)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60)) 
    torchsummary.summary(model,(3,228,228))
    torch.save(model, "MyNet.pt")