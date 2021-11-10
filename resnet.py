import torch 
import torch.nn as nn 
from augmentation.USM import USM2d 


class Resnet18(nn.Module):
	def __init__(self, input_size = (256, 256), N_class = 5):
		super().__init__()
		self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
		self.model.fc = torch.nn.Linear(in_features = 512, out_features = N_class, bias = True)
		#self.model.activatef = torch.nn.Sigmoid()

	def forward(self, x):
		return self.model(x)




class Resnet18_iq(nn.Module):
	def __init__(self, input_size = (256, 256), N_class = 5):
		super().__init__()
		self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
		self.model.fc = torch.nn.Linear(in_features = 512, out_features = N_class, bias = True)
		#self.model.activatef = torch.nn.Sigmoid()

		self.initial_layer = USM2d(3,3)

	def forward(self, x):
		x = self.initial_layer(x)

		return self.model(x)