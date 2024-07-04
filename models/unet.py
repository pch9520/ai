#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from models.cnn import Down, Up, DoubleConv


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # left
        self.left_conv_1 = DoubleConv(3, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)
        self.down_1 = Down()
        self.down_2 = Down()
        self.down_3 = Down()
        self.down_4 = Down()

        # center
        self.center_conv = DoubleConv(512, 1024)

        # right
        self.up_1 = Up(1024, 512)
        self.up_2 = Up(512, 256)
        self.up_3 = Up(256, 128)
        self.up_4 = Up(128, 64)

        self.output = nn.Conv2d(64, 21, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)
        x = self.center_conv(x4_down)
        x = self.up_1(x, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        return self.softmax(self.output(x))


if __name__ == '__main__':
    model = UNet()
    from utils.utils import weight_init
    weight_init(model)
    a = torch.rand(10, 3, 32, 32)
    b = model(a)
    print(b.size())
