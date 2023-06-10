
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
    

def run():

    import utils as U

    device = U.get_device()
    print(device)

    batch_size = 512
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_transforms = U.get_train_transforms()
    test_transforms = U.get_test_transforms()

    train_dataset = U.get_train_dataset(train_transforms)
    test_dataset = U.get_test_dataset(test_transforms)

    train_dataloader = U.get_train_dataloader(train_dataset, **kwargs)
    test_dataloader = U.get_test_dataloader(test_dataset, **kwargs)

    U.show_image_grid(train_dataloader)

    model = Net().to('cpu')

    U.show_summary(model, -1, device.type)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    num_epochs = 20

    metrics = { 'train_acc': [], 'train_losses': [],
                'test_acc': [], 'test_losses': [] }

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        U.train(model, device, train_dataloader, optimizer, metrics)
        U.test(model, device, test_dataloader, metrics)
        scheduler.step()

    U.display_results(metrics)


if __name__ == '__main__':
    run()