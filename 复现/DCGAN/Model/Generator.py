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
        self.img_size = img_size
        self.denseLayer1 = nn.Linear(in_features=input_shape, out_features=256*(img_size[0]//4)*(img_size[1]//4))
        self.conv2dT1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.leakyRelu1 = nn.LeakyReLU(0.01)
        self.conv2dT2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.leakyRelu2 = nn.LeakyReLU(0.01)
        self.conv2dT3 = nn.ConvTranspose2d(in_channels=64, out_channels=img_size[-1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.tanh = nn.Tanh()

        self.setOptimizer(lr=0.0003)
    
    def forward(self, x):
        x = self.denseLayer1(x)
        x = x.reshape(-1,256,self.img_size[0]//4,self.img_size[1]//4)
        x = self.conv2dT1(x)
        x = self.batchnorm1(x)
        x = self.leakyRelu1(x)
        x = self.conv2dT2(x)
        x = self.batchnorm2(x)
        x = self.leakyRelu2(x)
        x = self.conv2dT3(x)
        x = self.tanh(x)
        return x
    
    def setOptimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def LossFunc(self):
        return nn.BCELoss().cuda()