
import cv2
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import warnings
from scipy.signal import argrelmin
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2

import torch

import torch.nn as nn
from torch.nn.functional import relu, sigmoid

def normalize(image):
    if len(image.shape)==2:
        m = np.mean(image)
        sd = np.std(image)
        if sd!=0:
            return (image-m)/sd
        else:
            return (image-m)
    elif len(image.shape)==3:
        m = np.mean(image)
        sd = np.std(image)
        if sd!=0:
            return (image-m)/sd
        else:
            return (image-m)
    else:
        return np.nan


class mach1(nn.Module):
    def __init__(self):
        super(mach1, self).__init__()
        self.conv0 = nn.Conv2d(3, 96, 11, stride=4)
        self.pool0 = nn.MaxPool2d(3, stride=2)
        self.conv1 = nn.Conv2d(96, 256, 5, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv3 = nn.Conv2d(384, 384, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.fc0 = nn.Linear(384 * 2 * 2, 42)
        self.fc1 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv0(x)
        # print(x.shape)
        x = self.pool0(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(-1, 384 * 2 * 2)
        x = self.fc0(x)
        return self.fc1(x)


class mach2(nn.Module):
    def __init__(self):
        super(mach2, self).__init__()
        self.conv0 = nn.Conv2d(3, 96, 11, stride=4)
        self.pool0 = nn.MaxPool2d(3, stride=2)
        self.conv1 = nn.Conv2d(96, 256, 5, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv3 = nn.Conv2d(384, 384, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.fc0 = nn.Linear(384 * 2 * 2, 42*42)
        self.fc1 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv0(x)
        # print(x.shape)
        x = self.pool0(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(-1, 384 * 2 * 2)
        x = self.fc0(x)
        return self.fc1(x)



model1 = mach1()
model2 = mach1()
model3 = mach2()

X = []
y = []
from cv2 import imread
import os.path
for path, directories, files in os.walk('encoded_samples/'):
     for file in files:
         image = imread(os.path.join(path, file))
         y_loc=(file.split("_")[0:2])
         y.append([int(y_loc[0]),int(y_loc[1]), 42*int(y_loc[0])+int(y_loc[1])])
         image = np.rollaxis(image, 2, 0)
         X.append([normalize(image)])

def train(X, y, models, lr=0.001, epochs=200):
    loss_function = nn.CrossEntropyLoss()
    losses1 = []
    losses2 = []
    losses3 = []
    model1 = models[0]
    model2 = models[1]
    model3 = models[2]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = Variable(torch.Tensor(X_train))
    X_test = Variable(torch.Tensor(X_test))
    y_1_train = Variable(torch.Tensor(y_train[:, 0]).long())
    y_1_test = Variable(torch.Tensor(y_test[:, 0]).long())
    y_2_train = Variable(torch.Tensor(y_train[:, 1]).long())
    y_2_test = Variable(torch.Tensor(y_test[:, 1]).long())
    y_3_train = Variable(torch.Tensor(y_train[:, 2]).long())
    y_3_test = Variable(torch.Tensor(y_test[:, 2]).long())
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
    for epoch in range(epochs):
        print("epoch:", epoch)
        out1 = model1(X_train)
        out2 = model2(X_train)
        out3 = model3(X_train)
        loss1 = loss_function(out1, y_1_train)
        loss2 = loss_function(out2, y_2_train)
        loss3 = loss_function(out3, y_3_train)
        print("loss1:", round(float(loss1.detach()),4), end=" ")
        print("loss2:", round(float(loss2.detach()),4), end=" ")
        print("loss3:", round(float(loss3.detach()),4))
        losses1.append(loss1.detach())
        losses2.append(loss2.detach())
        losses3.append(loss3.detach())
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        loss1.backward()
        loss2.backward()
        loss3.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        print("acc1:", round(float(torch.mean((torch.argmax(model1(X_test),dim=1)==y_1_test).float()))*100,2),"%",end=" ")
        print("acc2:", round(float(torch.mean((torch.argmax(model2(X_test),dim=1)==y_2_test).float()))*100,2),"%",end=" ")
        print("acc3:", round(float(torch.mean((torch.argmax(model3(X_test),dim=1)==y_3_test).float()))*100,2),"%")
        print("acc1:", round(float(torch.mean((torch.argmax(model1(X_train), dim=1) == y_1_train).float())) * 100, 2),
              "%", end=" ")
        print("acc2:", round(float(torch.mean((torch.argmax(model2(X_train), dim=1) == y_2_train).float())) * 100, 2),
              "%", end=" ")
        print("acc3:", round(float(torch.mean((torch.argmax(model3(X_train), dim=1) == y_3_train).float())) * 100, 2),
              "%")
    torch.save(model1.state_dict(), "model1_"+str(epochs)+"_epochs.d")
    torch.save(model1, "model1_"+str(epochs)+"_epochs.m")
    torch.save(model2.state_dict(), "model2_"+str(epochs)+"_epochs.d")
    torch.save(model2, "model2._"+str(epochs)+"_epochsm")
    torch.save(model3.state_dict(), "model3_"+str(epochs)+"_epochs.d")
    torch.save(model3, "model3_"+str(epochs)+"_epochs.m")

X = np.concatenate(X)
y = np.array(y)
model1 = mach1()
model2 = mach1()
model3 = mach2()
models = [model1,model2,model3]
train(X,y,models,lr=0.001,epochs=200)