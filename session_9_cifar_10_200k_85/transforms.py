from torchvision import transforms

def get_train_transforms():
    '''
    This function loads the train transforms

    Usage:
    >>> train_transforms = get_train_transforms()
    '''
    # Train data transformations
    train_transforms = transforms.Compose([
                                        # transforms.RandomRotation((-7.0, 7.0), fill=(1, 1, 1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
                                                             (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                                        ])
    return train_transforms

def get_test_transforms():
    '''
    This function loads the test transforms

    Usage:
    >>> test_transforms = get_test_transforms()
    '''
    # Test data transformations
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
                                                             (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                                        ])

    return test_transforms