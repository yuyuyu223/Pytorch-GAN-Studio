'''
Date: 2021-11-25 21:50:04
LastEditors: HowsenFisher
LastEditTime: 2021-11-25 23:06:54
FilePath: \GAN\Model\Generator.py
'''
import torch
from torch import nn

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)

class Generator(nn.Module):
    def __init__(self, input_shape, img_size):
        super(Generator, self).__init__()
        input_size = 1
        for i in input_shape:
            input_size *= i
        self.denseLayer1 = nn.Linear(in_features=input_size, out_features=128)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.denseLayer2 = nn.Linear(in_features=128, out_features=img_size[0]*img_size[1]*img_size[2])
        self.reshapeLayer = View(img_size)
    
    def forward(self, x):
        x = self.denseLayer1(x)
        x = self.relu(x)
        x = self.denseLayer2(x)
        x = self.reshapeLayer(x)
        return x

