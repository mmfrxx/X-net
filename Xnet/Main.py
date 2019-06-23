import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from glob import glob
from itertools import chain
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
import torch.optim as optim
from ChestDataset import ChestDataset
from Xnet import Xnet

number_of_classes = 13
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
               'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
IMG_DIR = "./images/"
CSV_FILE = "sample_labels.csv"
batch_size = 64

all_xray_df = pd.read_csv(CSV_FILE)
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join(IMG_DIR, '*.png'))}
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x) > 0]

print('All Labels ({}): {}'.format(len(all_labels), all_labels))

for c_label in all_labels:
    if len(c_label) > 1:  # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

MIN_CASES = 20   # delete data with less amount
all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum() > MIN_CASES]
all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
print(len(all_xray_df['path']))

train_x, test_x, train_y, test_y = train_test_split(all_xray_df['path'].to_numpy(),
                                                    all_xray_df['disease_vec'].to_numpy(),
                                                    test_size=0.40,
                                                    random_state=2018)
test_x, val_x, test_y, val_y = train_test_split(test_x, test_y,
                                                test_size=0.50,
                                                random_state=2018)

test = ChestDataset(image_names=test_x, labels=test_y,
                    transform=transforms.Compose([transforms.Scale(256),
                                                  transforms.Grayscale(), transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485], std=[0.229])]))
test_loader = DataLoader(test, batch_size=1, shuffle=True)

train = ChestDataset(train_x, train_y,
                     transform=transforms.Compose([transforms.Scale(256),
                                                   transforms.Grayscale(), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485], std=[0.229])]))
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

valid = ChestDataset(val_x, val_y,
                     transform=transforms.Compose([transforms.Scale(256),
                                                   transforms.Grayscale(), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485], std=[0.229])]))

valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

model = Xnet(number_of_classes)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr = 1e-4,weight_decay = 0.005)

EPOCHS = 10
best_acc = 0.0
best_model =  model.state_dict()

for epoch in range(EPOCHS):
  r_loss = 0
  train_correct = 0
  r_total = 0 
  model.train()
  for data in train_loader:
    
    inputs = data['image'].to(device)
    labels= data['label'].to(device).squeeze(dim = 2)
    outputs = model(inputs)
    
    optimizer.zero_grad()
    labels = labels.type(torch.FloatTensor)
    outputs =outputs.type(torch.FloatTensor)
    
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    
    outputs = torch.round(outputs)
    
    labels = labels.type(torch.LongTensor)
    outputs =outputs.type(torch.LongTensor)
    
    train_correct += torch.sum(outputs == labels).item()
    r_loss += loss.item()
    r_total += labels.size(0)
  
  with torch.no_grad():
    val_losses = 0
    val_correct = 0
    model.eval()
    total = 0
    
    for data in valid_loader:
      inputs = data['image'].to(device)
      labels= data['label'].to(device).squeeze(dim = 2)
      outputs = model(inputs)
      
      labels = labels.type(torch.FloatTensor)
      outputs =outputs.type(torch.FloatTensor)
      
      val_loss = criterion(outputs,labels)
      val_losses += val_loss.item()
      
      outputs = torch.round(outputs)
      
      labels = labels.type(torch.LongTensor)
      outputs =outputs.type(torch.LongTensor)
      val_correct += torch.sum(outputs ==labels).item()
      total += labels.size(0)
      
  r_loss = r_loss/total
  train_correct = train_correct/r_total/number_of_classes
  
  val_loss = val_losses/ total
  val_correct = val_correct/ total/number_of_classes
  
  if val_correct > best_acc:
    bect_acc = val_correct
    best_model = model.state_dict()
  
       
  print("Epoch:" + str(epoch) +". Training loss:{: .3f}.  Train correct: {: .3f}".format(r_loss, train_correct))
  print('Validation loss: {: .3f}, {: .3f}'.format(val_loss, val_correct))

#SAVE THE STATE
torch.save(best_model, "./models/model" + str(EPOCHS) + ".pth")

#Testing process
with torch.no_grad():
  correct = 0
  total = 0
  for data in test_loader:
    inputs = data['image'].to(device)
    labels= data['label'].to(device).squeeze(dim = 2)
    outputs = model(inputs)
    outputs = torch.round(outputs)
    
    labels = labels.type(torch.LongTensor)
    outputs =outputs.type(torch.LongTensor)
    
    total +=labels.size(0)
    correct +=torch.sum(outputs == labels).item()
print("Accuracy of the network is %f %%" %(correct*100/number_of_classes/ total))
