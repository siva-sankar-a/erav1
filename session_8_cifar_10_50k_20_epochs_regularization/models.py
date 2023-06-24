
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net_CIFAR10_BatchNorm(nn.Module):

    '''
    This class implements the neural network model

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    '''

    #This defines the structure of the NN.
    def __init__(self):
        super(Net_CIFAR10_BatchNorm, self).__init__()
        self.conv1 = self.conv3x3_bn_dropout(3, 8, padding=1)
        self.conv2 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv3 = self.conv1x1(8, 8, padding=1)

        self.pool1 = self.conv3x3_bn_dropout(8, 8, stride=2)
        self.downsample1 = self.conv3x3_bn_dropout(3, 8, padding=1, stride=2)

        self.conv4 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv5 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv6 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv7 = self.conv1x1(8, 8, padding=1)

        self.pool2 = self.conv3x3_bn_dropout(8, 8, stride=2)
        self.downsample2 = self.conv3x3_bn_dropout(8, 8, padding=1, stride=2)

        self.conv8 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv9 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv10 = self.conv3x3_bn_dropout(8, 8, padding=1)

        self.gap = nn.AvgPool2d(8)
        self.mixer = nn.Sequential(
            self.conv1x1(8, 10),
        )


    def conv3x3_bn_dropout(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dropout=0.0):
          if dropout:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout),
                )
          else:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                )

    def conv3x3(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
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
        x = self.conv2(x)
        x = self.conv3(x)
        # pooling
        x = self.pool1(x)

        # residual connection
        i1 = self.downsample1(i1)
        x = x + i1

        # Block 2
        i2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # pooling
        x = self.pool2(x)

        # residual connection
        i2 = self.downsample2(i2)
        x = x + i2

        # Block 3
        i3 = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # residual connection
        x = x + i3

        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)

class Net_CIFAR10_GroupNorm(nn.Module):

    '''
    This class implements the neural network model

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    '''

    #This defines the structure of the NN.
    def __init__(self):
        super(Net_CIFAR10_GroupNorm, self).__init__()
        self.conv1 = self.conv3x3_bn_dropout(3, 8, padding=1)
        self.conv2 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv3 = self.conv1x1(8, 8, padding=1)

        self.pool1 = self.conv3x3_bn_dropout(8, 16, stride=2)
        self.downsample1 = self.conv3x3_bn_dropout(3, 16, padding=1, stride=2)

        self.conv4 = self.conv3x3_bn_dropout(16, 16, padding=1)
        self.conv5 = self.conv3x3_bn_dropout(16, 16, padding=1)
        self.conv6 = self.conv3x3_bn_dropout(16, 16, padding=1)
        self.conv7 = self.conv1x1(16, 16, padding=1)

        self.pool2 = self.conv3x3_bn_dropout(16, 32, stride=2)
        self.downsample2 = self.conv3x3_bn_dropout(16, 32, padding=1, stride=2)

        self.conv8 = self.conv3x3_bn_dropout(32, 32, padding=1)
        self.conv9 = self.conv3x3_bn_dropout(32, 32, padding=1)
        self.conv10 = self.conv3x3_bn_dropout(32, 32, padding=1)

        self.gap = nn.AvgPool2d(8)
        self.mixer = nn.Sequential(
            self.conv1x1(32, 10),
        )


    def conv3x3_bn_dropout(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dropout=0.1):
          if dropout:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.GroupNorm(4, out_channels),
                nn.Dropout(dropout),
                )
          else:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.GroupNorm(4, out_channels),
                )

    def conv3x3(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
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
        x = self.conv2(x)
        x = self.conv3(x)
        # pooling
        x = self.pool1(x)

        # residual connection
        i1 = self.downsample1(i1)
        x = x + i1

        # Block 2
        i2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # pooling
        x = self.pool2(x)

        # residual connection
        i2 = self.downsample2(i2)
        x = x + i2

        # Block 3
        i3 = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # residual connection
        x = x + i3

        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)

class Net_CIFAR10_LayerNorm(nn.Module):

    '''
    This class implements the neural network model

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    '''

    #This defines the structure of the NN.
    def __init__(self):
        super(Net_CIFAR10_LayerNorm, self).__init__()
        self.conv1 = self.conv3x3_bn_dropout(3, 8, padding=1)
        self.conv2 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv3 = self.conv1x1(8, 8, padding=1)

        self.pool1 = self.conv3x3_bn_dropout(8, 16, stride=2)
        self.downsample1 = self.conv3x3_bn_dropout(3, 16, padding=1, stride=2)

        self.conv4 = self.conv3x3_bn_dropout(16, 16, padding=1)
        self.conv5 = self.conv3x3_bn_dropout(16, 16, padding=1)
        self.conv6 = self.conv3x3_bn_dropout(16, 16, padding=1)
        self.conv7 = self.conv1x1(16, 16, padding=1)

        self.pool2 = self.conv3x3_bn_dropout(16, 32, stride=2)
        self.downsample2 = self.conv3x3_bn_dropout(16, 32, padding=1, stride=2)

        self.conv8 = self.conv3x3_bn_dropout(32, 32, padding=1)
        self.conv9 = self.conv3x3_bn_dropout(32, 32, padding=1)
        self.conv10 = self.conv3x3_bn_dropout(32, 32, padding=1)

        self.gap = nn.AvgPool2d(8)
        self.mixer = nn.Sequential(
            self.conv1x1(32, 10),
        )


    def conv3x3_bn_dropout(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dropout=0.0):
          if dropout:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.GroupNorm(1, out_channels),
                nn.Dropout(dropout),
                )
          else:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                 nn.GroupNorm(1, out_channels),
                )

    def conv3x3(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
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
        x = self.conv2(x)
        x = self.conv3(x)
        # pooling
        x = self.pool1(x)

        # residual connection
        i1 = self.downsample1(i1)
        x = x + i1

        # Block 2
        i2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # pooling
        x = self.pool2(x)

        # residual connection
        i2 = self.downsample2(i2)
        x = x + i2

        # Block 3
        i3 = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # residual connection
        x = x + i3

        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)

class Net_MNIST_4k(nn.Module):

    '''
    This class implements the neural network model

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    '''

    #This defines the structure of the NN.
    def __init__(self):
        super(Net_MNIST_4k, self).__init__()
        self.conv1 = self.conv3x3_bn_dropout(1, 8, padding=1)
        self.conv2 = self.conv3x3_bn_dropout(8, 8)

        # self.squeeze1 = self.conv1x1(4, 1)

        self.conv3 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv4 = self.conv3x3_bn_dropout(8, 8)
        self.conv5 = self.conv3x3_bn_dropout(8, 8)
        self.conv6 = self.conv3x3_bn_dropout(8, 8, stride=2)

        # self.squeeze2 = self.conv1x1(8, 1)

        self.conv7 = self.conv3x3_bn_dropout(8, 8, padding=1)
        self.conv8 = self.conv3x3_bn_dropout(8, 8)
        self.conv9 = self.conv3x3_bn_dropout(8, 16)
        self.conv10 = self.conv3x3_bn_dropout(16, 16)

        self.gap = nn.AvgPool2d(4)
        self.mixer = nn.Sequential(
            self.conv1x1(16, 10),
        )


    def conv3x3_bn_dropout(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dropout=0.1):
          if dropout:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout),
                )
          else:
            return nn.Sequential(
                self.conv3x3(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                )

    def conv3x3(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)

    def conv1x1(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.squeeze1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.squeeze2(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

class Net_MNIST_17k(nn.Module):

    '''
    This class implements the neural network model 

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    '''

    #This defines the structure of the NN.
    def __init__(self):
        super(Net_MNIST_17k, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(1, 2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            self.conv3x3(2, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            self.conv3x3(2, 16, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            self.conv3x3(16, 16, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.conv5 = nn.Sequential(
            self.conv3x3(16, 16, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.conv6 = nn.Sequential(
            self.conv3x3(16, 16, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            self.conv3x3(16, 16, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.conv8 = nn.Sequential(
            self.conv3x3(16, 16, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        self.conv9 = nn.Sequential(
            self.conv3x3(16, 32, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )
        self.gap = nn.AvgPool2d(4)
        self.mixer = nn.Sequential(
            self.conv1x1(32, 10),
        )


    def conv3x3(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)

    def conv1x1(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
          return nn.Conv2d(in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.maxpool2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)