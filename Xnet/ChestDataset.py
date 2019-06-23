import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform, io
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from itertools import chain
import warnings
warnings.filterwarnings("ignore")
import torch.backends.cudnn as cudnn

class ChestDataset(Dataset):
  def __init__(self, image_names, labels, transform):
    img_names = []
    for name in image_names:
      img_names.append(name)
    self.image_names = img_names
    self.labels = labels
    self.transform = transform
    
  def __getitem__(self,index):
    image_name = self.image_names[index]
    image = Image.open(image_name).convert('L')
    label = self.labels[index]
    if self.transform:
      image = self.transform(image)
    sample = {'image': image, 'label': torch.tensor(np.vstack(label).astype(np.float), dtype = torch.float32)} 
    return sample
  
  
  def __len__(self):
    return len(self.labels)
