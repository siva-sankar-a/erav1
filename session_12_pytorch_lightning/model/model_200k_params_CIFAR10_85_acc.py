
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        # C1
        self.conv1 = self.conv_block_3x3(3, 32)
        self.downsample1 = self.conv3x3_bn_dropout(3, 32, padding=1, stride=2)

        # C2
        self.conv2 = self.conv_block_3x3(32, 32, depthwise_seperable=True)
        self.downsample2 = self.conv3x3_bn_dropout(32, 32, padding=1, stride=2)

        #C3
        self.conv3 = self.conv_block_3x3(32, 32)
        self.downsample3 = self.conv3x3_bn_dropout(32, 32, padding=1, stride=2)

        #C4
        self.conv4 = self.conv_block_3x3(32, 64, pool=False)
        self.downsample4 = self.conv3x3_bn_dropout(32, 64, padding=1)

        self.gap = nn.AvgPool2d(4)
        self.mixer = nn.Sequential(
            self.conv1x1(64, 10),
        )

    def conv_block_3x3(self, in_channels, out_channels, pool=True, depthwise_seperable=False):
        if pool:
            return nn.Sequential(
                self.conv3x3_bn_dropout(in_channels, out_channels, padding=1),
                self.conv3x3_bn_dropout(out_channels, out_channels, padding=1, groups=(out_channels if depthwise_seperable else 1)), # Depthwise seperable convolution
                self.conv3x3_bn_dropout(out_channels, out_channels, dilation=2, padding=2, stride=2) # Dilated convlution
            )
        else:
            return nn.Sequential(
                self.conv3x3_bn_dropout(in_channels, out_channels, padding=1),
                self.conv3x3_bn_dropout(out_channels, out_channels, padding=1),
                self.conv3x3_bn_dropout(out_channels, out_channels, padding=1)
        )
    
    def conv3x3_bn_dropout(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, dropout=0.0):
          if dropout:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout),
                )
          else:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                nn.ReLU(),
                 nn.BatchNorm2d(out_channels),
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

        # Block 1
        i1 = x
        x = self.conv1(x)
        i1 = self.downsample1(i1)
        x = x + i1

        # Block 2
        i2 = x
        x = self.conv2(x)
        i2 = self.downsample2(i2)
        x = x + i2

        # Block 3
        i3 = x
        x = self.conv3(x)
        i3 = self.downsample3(i3)
        x = x + i3

        # Block 4
        i4 = x
        x = self.conv4(x)
        i4 = self.downsample4(i4)
        x = x + i4

        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)

        # return x