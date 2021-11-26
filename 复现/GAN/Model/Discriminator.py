'''
Date: 2021-11-25 22:40:36
LastEditors: HowsenFisher
LastEditTime: 2021-11-25 23:16:47
FilePath: \GAN\Model\Discriminator.py
'''
import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten(start_dim=len(input_shape), end_dim=1)
        res = 1
        for i in input_shape:
            res *= i
        self.denseLayer1 = nn.Linear(in_features=res, out_features=128)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.denseLayer2 = nn.Linear(in_features=128, out_features=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.denseLayer1(x)
        x = self.relu(x)
        x = self.denseLayer2(x)
        x = self.sigmod(x)
        return x