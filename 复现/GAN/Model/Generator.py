'''
Date: 2021-11-25 21:50:04
LastEditors: HowsenFisher
LastEditTime: 2021-11-27 20:04:57
FilePath: \GAN\Model\Generator.py
'''
import torch
from torch import nn
import torch.optim as optim

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)

class Generator(nn.Module):
    def __init__(self, input_shape, img_size):
        super(Generator, self).__init__()
        self.denseLayer1 = nn.Linear(in_features=input_shape, out_features=256)
        self.relu1 = nn.ReLU()
        self.denseLayer2 = nn.Linear(in_features=256, out_features=256)
        self.relu2 = nn.ReLU()
        self.denseLayer3 = nn.Linear(in_features=256, out_features=img_size[0]*img_size[1]*img_size[2])
        self.tanh = nn.Tanh()

        self.setOptimizer(lr=0.0003)
    
    def forward(self, x):
        x = self.denseLayer1(x)
        x = self.relu1(x)
        x = self.denseLayer2(x)
        x = self.relu2(x)
        x = self.denseLayer3(x)
        x = self.tanh(x)
        x = x.reshape(-1,1,28,28)
        return x
    
    def setOptimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def LossFunc(self):
        return nn.BCELoss().cuda()