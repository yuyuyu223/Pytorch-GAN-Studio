'''
Date: 2021-11-25 22:40:43
LastEditors: HowsenFisher
LastEditTime: 2021-11-25 23:20:39
FilePath: \GAN\Model\GAN.py
'''
import torch
from torch import nn
from Model.Generator import Generator
from Model.Discriminator import Discriminator

class GAN(nn.Module):
    def __init__(self, input_shape, img_size):
        super(GAN, self).__init__()
        self.generator = Generator(input_shape, img_size)
        self.discriminator = Discriminator(img_size)
    
    def forward(self, x):
        x = self.generator.forward(x)
        x = self.discriminator.forward(x)
        return x