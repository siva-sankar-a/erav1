import torch
from torchvision import datasets

def get_train_dataset(train_transforms):
    '''
    This function loads the train dataset

    Usage:
    >>> train_transforms = get_train_transforms()
    >>> train_dataset = get_train_dataset(train_transforms)
    '''
    train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)
    return train_data

def get_test_dataset(test_transforms):
    '''
    This function loads the test dataset

    Usage:
    >>> test_transforms = get_test_transforms()
    >>> test_dataset = get_test_dataset(test_transforms)
    '''
    test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)
    return test_data


def get_train_dataloader(train_data, **kwargs):

    '''
    This function loads train dataloader

    Usage:
    >>> train_transforms = get_train_transforms()
    >>> train_dataset = get_train_dataset(train_transforms)
    >>> train_dataloader = get_train_dataloader(train_dataset, **kwargs)
    '''
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    return train_loader

def get_test_dataloader(test_data, **kwargs):

    '''
    This function loads test dataloader

    Usage:
    >>> test_transforms = get_test_transforms()
    >>> test_dataset = get_test_dataset(test_transforms)
    >>> test_dataloader = get_test_dataloader(test_dataset, **kwargs)
    '''
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    return test_loader


class CIFAR10_Dataset(datasets.CIFAR10):
    '''
    This class takes care of Albumentations Library in __getitem__
    '''
    def __init__(self, root="./data/", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    

def get_CIFAR10_albumentations_train_dataset(train_transforms):
    '''
    This function loads the train dataset

    Usage:
    >>> train_transforms = get_train_transforms()
    >>> train_dataset = get_train_dataset(train_transforms)
    '''
    train_data = CIFAR10_Dataset('../data', train=True, download=True, transform=train_transforms)
    return train_data

def get_CIFAR10_albumentations_test_dataset(test_transforms):
    '''
    This function loads the test dataset

    Usage:
    >>> test_transforms = get_test_transforms()
    >>> test_dataset = get_test_dataset(test_transforms)
    '''
    test_data = CIFAR10_Dataset('../data', train=False, download=True, transform=test_transforms)
    return test_data