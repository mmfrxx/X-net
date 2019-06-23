import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms,utils
from itertools import chain
import warnings
warnings.filterwarnings("ignore")
import torch.backends.cudnn as cudnn


# DEFINE THE NETWORK
class Xnet(nn.Module):
    def __init__(self,number_of_classes):
        super(Xnet, self).__init__()
				self.conv1 = nn.Conv2d(1,64,kernel_size = 3, padding = 1)                    
				self.batch_norm1 = nn.BatchNorm2d(64)                                                                                   
				self.pool = nn.MaxPool2d(2,2)                                             
				self.conv2 = nn.Conv2d(64,128, kernel_size =3, padding =1)                 
				self.batch_norm2 = nn.BatchNorm2d(128)                                       
				self.conv3 = nn.Conv2d(128,256, kernel_size = 3, padding = 1)                 
				self.batch_norm3 = nn.BatchNorm2d(256) 
				self.conv4 = nn.Conv2d(256,512, kernel_size = 3, padding = 1)
				self.batch_norm4 = nn.BatchNorm2d(512) 
				self.conv5 = nn.Conv2d(512,512, kernel_size = 3, padding = 1)
				self.conv6 = nn.Conv2d(512,256, kernel_size = 3, padding = 1)
				self.conv7 = nn.Conv2d(256,128, kernel_size = 3, padding = 1)
				self.conv8 = nn.Conv2d(128,128,kernel_size = 3, padding = 1)
				self.conv9 = nn.Conv2d(128,256,kernel_size = 3, padding = 1)
				self.conv10 = nn.Conv2d(256, 512, kernel_size = 3, padding =1)
				self.conv11 = nn.Conv2d(512,512, kernel_size = 3, padding = 1)
				self.conv12 = nn.Conv2d(512,256, kernel_size = 3, padding = 1)
				self.conv13 = nn.Conv2d(256,128, kernel_size = 3, padding = 1)
				self.conv14 = nn.Conv2d(128,64, kernel_size =3, padding = 1)
				self.conv15 = nn.Conv2d(64, number_of_classes, kernel_size = 1)
				self.fc1 = nn.Linear(13*256*256, 64)
				self.fc2 = nn.Linear(64, number_of_classes)
				
    def forward(self, x):
				act1 = F.relu(self.batch_norm1(self.conv1(x)))          #64 x 256 x 256
				x = self.pool(act1)                                     #64 x 128 x 128
				act2 = F.relu(self.batch_norm2(self.conv2(x)))          #128 x 128 x 128
				x = self.pool(act2)                                     #128 x 64 x 64
				act_3 = F.relu(self.batch_norm3(self.conv3(x)))         #256 x 64 x 64
				x = self.pool(act_3)                                    #256 x 32 x 32
				x = F.relu(self.batch_norm4(self.conv4(x)))   					#512 x 32 x 32
				x = F.relu(self.batch_norm4(self.conv5(x)))   					#512 x 32 x 32
				x = F.upsample(x, size = 64)   
				x = F.relu(self.batch_norm3(self.conv6(x)))  					  #256 x 64 x 64    
				x = x.add(act_3)    
				x = F.upsample(x, size = 128)                 					#256 x 128 x 128
				x = F.relu(self.batch_norm3(self.conv7(x)))   					#128 x 128 x 128
				x = x.add(act2)    
				act_8 = F.relu(self.batch_norm2(self.conv8(x))) 				#128 x 128 x 128
				x = self.pool(act_8)                            				#128 x 64 x 64
				act_9 = F.relu(self.batch_norm3(self.conv9(x))) 				#256 x 64 x 64
				x = self.pool(act_9)                            				#256 x 32 x 32
				x = F.relu(self.batch_norm4(self.conv10(x)))    				#512 x 32 x 32
				x = F.relu(self.batch_norm4(self.conv11(x)))   					#512 x 32 x 32    
				x = F.upsample(x, size = 64)                    				#512 x 64 x 64
				x = F.relu(self.batch_norm3(self.conv12(x)))    				#256 x 64 x 64
				x = x.add(act_9)    
				x = F.upsample(x, size = 128)                   				#256 x 128 x 128 
				x = F.relu(self.batch_norm2(self.conv13(x)))    				#128 x 128 x 128
				x = x.add(act_8)    
				x = F.upsample(x, size = 256)                   				#128 x 256 x 256
				x = F.relu(self.batch_norm1(self.conv14(x)))    				#64 x 256 x 256
				x = x.add(act1)
				x = self.conv15(x)                              				#13 x 256 x 256
				x = x.view(-1,13*256*256)
				x = F.relu(self.fc1(x))
				x = nn.Sigmoid(self.fc2(x))
				return x
