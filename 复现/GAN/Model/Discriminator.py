'''
Date: 2021-11-25 22:40:36
LastEditors: HowsenFisher
LastEditTime: 2021-11-27 20:05:47
FilePath: \GAN\Model\Discriminator.py
'''
import torch
from torch import nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        res = 1
        for i in input_shape:
            res *= i
        self.denseLayer1 = nn.Linear(in_features=res, out_features=256)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.denseLayer2 = nn.Linear(in_features=256, out_features=256)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.denseLayer3 = nn.Linear(in_features=256, out_features=1)
        
        self.sigmod = nn.Sigmoid()
        self.setOptimizer(lr=0.0003)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.denseLayer1(x)
        x = self.relu1(x)
        x = self.denseLayer2(x)
        x = self.relu2(x)
        x = self.denseLayer3(x)
        x = self.sigmod(x)
        return x
    
    def setOptimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def LossFunc(self):
        return nn.BCELoss().cuda()
