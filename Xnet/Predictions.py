import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from Xnet import Xnet
import torch
from PIL import Image
from torchvision import transforms, utils

number_of_classes= 13
model = Xnet(number_of_classes)

model.load_state_dict(torch.load(PATH TO YOUR MODEL STATE DICT))
model.eval()
input = PATH TO YOUR IMAGE

image = Image.open(input).convert('L')
transform = transforms.Compose([transforms.Scale(256),
                                                   transforms.Grayscale(), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485], std=[0.229])])
image = transform(image)
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
prediction = model(image)
prediction = torch.round(prediction)
print("Predicted:\n")
for i in range(number_of_classes):
	if prediction[i].item() == 1:
		print(CLASS_NAMES[i])
	


