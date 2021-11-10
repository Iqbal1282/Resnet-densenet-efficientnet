import math
import torch
import torch.nn as nn
from torch.nn.functional import conv2d
import torch.nn.functional as F



#The kernel_size, var_x, var_y all should be torch_tensors
def gaussian_kernel(kernel_size, var_x, var_y):
    ax = torch.round(torch.linspace(-math.floor(kernel_size/2), math.floor(kernel_size/2),
                                    kernel_size), out=torch.FloatTensor())
    x = ax.view(1, -1).repeat(ax.size(0), 1).to('cuda')
    y = ax.view(-1, 1).repeat(1, ax.size(0)).to('cuda')
    x2 = torch.pow(x,2)
    y2 = torch.pow(y,2)
    std_x = torch.pow(var_x, 0.5)
    std_y = torch.pow(var_y, 0.5)
    temp = - ((x2/var_x) + (y2/var_y)) / 2
    kernel = torch.exp(temp)/(2*math.pi*std_x*std_y)
    kernel = kernel/kernel.sum()
    return kernel
    
class GB2d_img(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(GB2d_img, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        #self.padding = padding
        #self.aug_train = aug_train
        
        # if self.aug_train:
        #     self.var_x = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        #     self.var_y = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        #     self.init_parameters()
        # else:
        #     self.var_x = torch.FloatTensor([var_x])
        #     self.var_y = torch.FloatTensor([var_y])
        #     self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
        #     self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

        self.layer1 = nn.Conv2d(self.in_channel, 32, (3,3),2) 
        self.layer2 = nn.MaxPool2d(3,2)
        self.layer3 = nn.Conv2d(32,64, (3,3), (1,1))
        self.layer4 = nn.MaxPool2d((3,3),(2,2))
        self.layer5 = nn.Conv2d(64,64, (3,3), (1,1))
        self.layer6 = nn.MaxPool2d((3,3),(2,2))
        self.layer7 = nn.Conv2d(64,32, (3,3), (1,1))
        self.layer8 = nn.MaxPool2d((3,3),(2,2))
        self.layer9 = nn.Linear(800,2)
            
    def init_parameters(self):
        self.var_x.data.uniform_(0, 1)
        self.var_y.data.uniform_(0, 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

        
    def forward(self, img):
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
        x = torch.sigmoid(self.layer9(x))
        #x = x.view(-1,2)
        
        output = torch.zeros_like(img)
        for i in range(batch_size):
            self.var_x, self.var_y = x[i]
            self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size) # 256x256
        
            output[i] = conv2d(img[i].view(1*self.in_channel,1,h,w), self.kernel, padding = self.padding).view(self.in_channel,h,w)



            # self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
            # self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            
        #output = conv2d(x.view(batch_size*self.in_channel,1,h,w), self.kernel, padding = self.padding)
        h_out = math.floor((h - self.kernel_size + 2*self.padding)+1)
        w_out = math.floor((w - self.kernel_size + 2*self.padding)+1)
        output = output.view(batch_size, self.in_channel, h_out, w_out)
        return output
    