'''
Date: 2021-11-25 22:40:43
LastEditors: HowsenFisher
LastEditTime: 2021-11-27 19:29:41
FilePath: \GAN\Model\GAN.py
'''
import torch
from torch import nn
from Model.Generator import Generator
from Model.Discriminator import Discriminator
from torchsummary import summary


class GAN(nn.Module):
    def __init__(self, input_shape, img_size):
        super(GAN, self).__init__()
        self.input_shape = input_shape
        self.generator = Generator(input_shape, img_size).cuda()
        self.discriminator = Discriminator(img_size).cuda()
    
    def forward(self, x):
        x = self.generator.forward(x)
        x = self.discriminator.forward(x)
        return x
    
    def PrintNet(self, logger):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("总参数:%d  训练参数:%d"%(total_num,trainable_num))
        summary(self,(self.input_shape,))
    

    
