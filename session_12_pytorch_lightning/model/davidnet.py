import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import pytorch_lightning as pl
import torchmetrics
import torch.optim as optim

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class Net(nn.Module):

    '''
    This class implements the neural network model

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    '''

    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        # Prep Layer
        self.prep = self.conv3x3_bn_dropout(3, 64, padding=1)

        # Layer 1
        self.conv_bn_pool_1 = self.conv3x3_maxpool(64, 128)
        # Layer 1 - residual
        self.res_1 = self.conv3x3_bn_dropout(128, 128, padding=1)
        self.res_2 = self.conv3x3_bn_dropout(128, 128, padding=1)
        self.downsample_1 = self.conv1x1(128, 128)

        # Layer 2
        self.conv_bn_pool_2 = self.conv3x3_maxpool(128, 256)

        # Layer 3
        self.conv_bn_pool_3 = self.conv3x3_maxpool(256, 512)
        # Layer 3 - residual
        self.res_3 = self.conv3x3_bn_dropout(512, 512, padding=1)
        self.res_4 = self.conv3x3_bn_dropout(512, 512, padding=1)
        self.downsample_2 = self.conv1x1(512, 512)

        # Pool
        self.pool = nn.MaxPool2d(4)

        # Linear
        self.linear = nn.Linear(512, 10)

    def conv3x3_maxpool(self, in_channels, out_channels):
        layers = [self.conv3x3_bn_dropout(in_channels, out_channels, padding=1),
                  nn.MaxPool2d(2)]

        return nn.Sequential(
                    *layers
                )

    def conv3x3_bn_dropout(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, dropout=None, is_relu_bn=True):

          if is_relu_bn:
            layers = [
                  self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(),
            ]
          else:
            layers = [
                  self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            ]

          if dropout:
            return nn.Sequential(
                    *layers,
                    nn.Dropout(dropout)
                )
          else:
            return nn.Sequential(
                    *layers
                )

    def conv3x3(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias)

    def conv1x1(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)


    def forward(self, x):

        # Prep Layer
        x = self.prep(x)

        # Layer 1
        x = self.conv_bn_pool_1(x)
        # Layer 1 - residual
        i = x
        x = self.res_1(x)
        x = self.res_2(x)
        x = x + i

        # Layer 2
        x = self.conv_bn_pool_2(x)

        # Layer 3
        x = self.conv_bn_pool_3(x)
        # Layer 3 - residual
        i = x
        x = self.res_3(x)
        x = self.res_4(x)
        x = x + i

        # Pool
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 512)

        # Linear
        x = self.linear(x)

        return x