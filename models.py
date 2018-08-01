## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        
#         self.conv3 = nn.Conv2d(64, 128, 5)
        
#         self.conv4 = nn.Conv2d(128, 256, 5)
        
#         self.conv5 = nn.Conv2d(256, 512, 5)
        
        self.fc1 = nn.Linear(53*53*64, 272)
        
        self.drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(272, 136)
        
        

        
    def forward(self, x):
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.tanh(self.conv1(x))) #110
        x = self.drop(x)
        x = self.pool(F.tanh(self.conv2(x))) #53
#         x = self.drop(x)
#         x = self.pool(F.relu(self.conv3(x))) #24
#         x = self.drop(x)
#         x = self.pool(F.relu(self.conv4(x))) #10
#         x = self.drop(x)
#         x = self.pool(F.relu(self.conv5(x))) #3
#         x = self.drop(x)
        #Flatten
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
