import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import random 
random.seed(0)



class ImageDataset(Dataset):
	def __init__(self, label_path, mode='train', image_shape = 256,subfolder = './'):
		self.subfolder = subfolder
		self._label_header = None
		self._image_paths = []
		self._labels = []
		self._mode = mode
		self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
					 {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
		with open(label_path) as f:
			header = f.readline().strip('\n').split(',')
			self._label_header = [
				header[7],
				header[10],
				header[11],
				header[13],
				header[15]]
			for line in f:
				labels = []
				fields = line.strip('\n').split(',')
				image_path = fields[0]
				flg_enhance = False
				for index, value in enumerate(fields[5:]):
					if index == 5 or index == 8:
						labels.append(self.dict[1].get(value))
					elif index == 2 or index == 6 or index == 10:
						labels.append(self.dict[0].get(value))
				# labels = ([self.dict.get(n, n) for n in fields[5:]])
				labels = list(map(int, labels))
				self._image_paths.append(image_path)
				assert os.path.exists(self.subfolder +image_path),self.subfolder+ image_path
				self._labels.append(labels)

		self._num_image = len(self._image_paths)
		self.image_shape = image_shape
		

	def __len__(self):
		return self._num_image

	def __getitem__(self, idx):
		assert os.path.exists(self.subfolder + self._image_paths[idx]),self.subfolder+ self._image_paths[idx]
		image = cv2.imread(self.subfolder + self._image_paths[idx], 0)
		image = cv2.resize(image, (self.image_shape,self.image_shape))
		image = Image.fromarray(image)
		image = np.array(image)
		image = self.transfrom(image)


		labels = np.array(self._labels[idx]).astype(np.float32)
		path = self._image_paths[idx]

		# if self._mode == 'train' or self._mode == 'dev':
		#     return (image, labels)
		# else:
		#     raise Exception('Unknown mode : {}'.format(self._mode))

		return (image, labels)

	def transfrom(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

		image = image.astype(np.float32) - 128.0 
		image /= 64.0 
		image = image.transpose((2, 0, 1))
		return image 


class ImageDataset_iq(Dataset):
	def __init__(self, label_path, mode='train', image_shape = 256,subfolder = './'):
		self.subfolder = subfolder
		self._label_header = None
		self._image_paths = []
		self._labels = []
		self._mode = mode
		self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
					 {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
		with open(label_path) as f:
			header = f.readline().strip('\n').split(',')
			self._label_header = [
				header[7],
				header[10],
				header[11],
				header[13],
				header[15]]
			for line in f:
				labels = []
				fields = line.strip('\n').split(',')
				image_path = fields[0]
				flg_enhance = False
				for index, value in enumerate(fields[5:]):
					if index == 5 or index == 8:
						if value == '-1.0':
							labels.append(str(np.float16(random.uniform(0, 1))))
						else:
							labels.append(self.dict[1].get(value))

					elif index == 2 or index == 6 or index == 10:
						#labels.append(self.dict[0].get(value))
						if value == '-1.0':
							labels.append(str(np.float16(random.uniform(0, 1))))
						else:
							labels.append(self.dict[0].get(value))

				# labels = ([self.dict.get(n, n) for n in fields[5:]])
				labels = list(map(float, labels))
				self._image_paths.append(image_path)
				assert os.path.exists(self.subfolder +image_path),self.subfolder+ image_path
				self._labels.append(labels)

		self._num_image = len(self._image_paths)
		self.image_shape = image_shape
		

	def __len__(self):
		return self._num_image

	def __getitem__(self, idx):
		assert os.path.exists(self.subfolder + self._image_paths[idx]),self.subfolder+ self._image_paths[idx]
		image = cv2.imread(self.subfolder + self._image_paths[idx], 0)
		image = cv2.resize(image, (self.image_shape,self.image_shape))
		image = Image.fromarray(image)
		image = np.array(image)
		image = self.transfrom(image)


		labels = np.array(self._labels[idx]).astype(np.float32)
		path = self._image_paths[idx]

		# if self._mode == 'train' or self._mode == 'dev':
		#     return (image, labels)
		# else:
		#     raise Exception('Unknown mode : {}'.format(self._mode))

		return (image, labels)

	def transfrom(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

		image = image.astype(np.float32) - 128.0 
		image /= 64.0 
		image = image.transpose((2, 0, 1))
		return image 


