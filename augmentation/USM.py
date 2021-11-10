# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:04:55 2020

@author: S. M. Jawwad Hossain
"""

import torch
from augmentation.GB import GB2d
import torch.nn as nn

class USM2d(nn.Module):
    def __init__(self, in_channel, kernel_size, var_x = None, var_y = None, alpha = None, aug_train = True):
        super(USM2d, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.aug_train = aug_train
        
        if self.aug_train:
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.var_x = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.var_y = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.init_parameters()
        else:
            self.alpha = torch.FloatTensor([alpha])
            self.var_x = torch.FloatTensor([var_x])
            self.var_y = torch.FloatTensor([var_y])
            self.GB_fcn = GB2d(in_channel, kernel_size, var_x, 
                               var_y, train = False, padding = self.padding)
    
    def init_parameters(self):
        self.alpha.data.uniform_(0,1)
        self.var_x.data.uniform_(0,1)
        self.var_y.data.uniform_(0,1)
        
    def forward(self, input_img):      
        if self.aug_train:
            self.GB_fcn = GB2d(self.in_channel, self.kernel_size, self.var_x, 
                               self.var_y, aug_train = False, padding = self.padding)
            blur_img = self.GB_fcn(input_img)
            usm_img = (1 + torch.tanh(self.alpha)) * input_img - torch.tanh(self.alpha) * blur_img
        else:
            blur_img = self.GB_fcn(input_img)
            usm_img = (1 + torch.tanh(self.alpha)) * input_img - torch.tanh(self.alpha) * blur_img
            
        return usm_img

