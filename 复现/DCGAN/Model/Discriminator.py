'''
Date: 2021-11-25 22:40:36
LastEditors: HowsenFisher
LastEditTime: 2021-11-27 20:05:47
FilePath: \GAN\Model\Discriminator.py
'''
import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        res = 1
        for i in input_shape:
            res *= i
        self.conv2d1 = nn.Conv2d(in_channels=input_shape[-1], out_channels=32, kernel_size=3, stride=2, padding=1)
        self.leakyRelu1 = nn.LeakyReLU(0.01)
        self.conv2d2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(num_features=64)
        self.leakyRelu2 = nn.LeakyReLU(0.01)
        self.conv2d3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.batchNorm2 = nn.BatchNorm2d(num_features=128)
        self.leakyRelu3 = nn.LeakyReLU(0.01)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features=11*11*128, out_features=1) 
        self.sigmod = nn.Sigmoid()
        self.setOptimizer(lr=0.0003)
    
    def forward(self, x):
        x = self.conv2d1(x)
        x = self.leakyRelu1(x)
        x = self.conv2d2(x)
        x = self.batchNorm1(x)
        x = self.leakyRelu2(x)
        x = self.conv2d3(x)
        x = self.batchNorm2(x)
        x = self.leakyRelu3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmod(x)
        return x
    
    def setOptimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def LossFunc(self):
        return nn.BCELoss().cuda()
