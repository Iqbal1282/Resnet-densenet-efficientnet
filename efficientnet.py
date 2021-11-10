from efficientnet_pytorch import EfficientNet
import torch 
import torch.nn as nn 
from augmentation.USM import USM2d 



class Efficientnet(nn.Module):
	def __init__(self, input_size = (256, 256), N_class = 5):
		super().__init__()
		self.model = EfficientNet.from_pretrained('efficientnet-b0')
		self.model._fc = torch.nn.Linear(in_features = 1280, out_features = N_class, bias = True)
		#self.model.activatef = torch.nn.Sigmoid()

	def forward(self, x):
		return self.model(x)


class Efficientnet_iq(nn.Module):
	def __init__(self, input_size = (256, 256), N_class = 5):
		super().__init__()
		self.model = EfficientNet.from_pretrained('efficientnet-b0')
		self.model._fc = torch.nn.Linear(in_features = 1280, out_features = N_class, bias = True)
		self.initial_layer = USM2d(3,3)

	def forward(self, x):
		x = self.initial_layer(x)
		#print(x.shape)
		return self.model(x)

