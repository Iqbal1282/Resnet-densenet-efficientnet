# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:04:55 202
"""

import torch
from augmentation.GB import GB2d
import torch.nn as nn
import torch.nn.functional as F

class USM2d_img(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(USM2d_img, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        
        
        self.layer1 = nn.Conv2d(self.in_channel, 32, (3,3),2) 
        self.layer2 = nn.MaxPool2d(3,2)
        self.layer3 = nn.Conv2d(32,64, (3,3), (1,1))
        self.layer4 = nn.MaxPool2d((3,3),(2,2))
        self.layer5 = nn.Conv2d(64,64, (3,3), (1,1))
        self.layer6 = nn.MaxPool2d((3,3),(2,2))
        self.layer7 = nn.Conv2d(64,32, (3,3), (1,1))
        self.layer8 = nn.MaxPool2d((3,3),(2,2))
        self.layer9 = nn.Linear(800,3)

    
    def init_parameters(self):
        self.alpha.data.uniform_(0,1)
        self.var_x.data.uniform_(0,1)
        self.var_y.data.uniform_(0,1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

        
    def forward(self, input_img):
        img = input_img   

        input_shape = img.shape
        batch_size, h, w = input_shape[0], input_shape[2], input_shape[3]
        
        #if self.aug_train:
        x = F.relu(self.layer1(img))
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
        x = F.relu(self.layer7(x))
        x = self.layer8(x)
        x = x.view(-1,self.num_flat_features(x))
        #print(x.shape)
        #print('--------------------------------------------------------')
        x = self.layer9(x)

        var_x = 100*torch.sigmoid(x[:,0])
        var_y = 100*torch.sigmoid(x[:,1])
        alpha = torch.tanh(x[:,2])

        batch_size = x.shape[0]

        my_output = torch.zeros_like(input_img)

        for i in range(batch_size):

            self.GB_fcn = GB2d(self.in_channel, self.kernel_size, var_x[i], 
                               var_y[i], aug_train = False, padding = self.padding)
            single_img  = input_img[i].unsqueeze(0)
            blur_img = self.GB_fcn(single_img)

            usm_img = (1 + torch.tanh(alpha[i]))*single_img  - torch.tanh(alpha[i]) * blur_img
            my_output[i] = usm_img.squeeze(0)
            
        return my_output 

