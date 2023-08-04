from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

def get_CIFAR10_train_transforms():
    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=25, p=0.5),
            A.CoarseDropout(max_holes=3, max_height=20, max_width=20, min_holes=1, min_height=4, min_width=4, 
                            fill_value=(0.4913997551666284, 0.48215855929893703, 0.446530913373161), 
                            mask_fill_value=None, always_apply=False, p=0.5),
            A.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.446530913373161), 
                        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
            ToTensorV2(),
        ]
    )

    return train_transforms

def get_CIFAR10_test_transforms():
    test_transforms = A.Compose(
        [
            A.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.446530913373161), 
                        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
            ToTensorV2(),
        ]
    )

    return test_transforms


def get_CIFAR10_DavidNet_train_transforms():
    train_transforms = A.Compose(
        [
            A.CropAndPad(px=4, pad_mode=4, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8,
                            fill_value=(0.4913997551666284, 0.48215855929893703, 0.446530913373161),
                            mask_fill_value=None, always_apply=False, p=0.5),
            A.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.446530913373161),
                        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
            ToTensorV2(),
        ]
    )

    return train_transforms

def get_CIFAR10_DavidNet_test_transforms():
    test_transforms = A.Compose(
        [
            A.Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.446530913373161),
                        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
            ToTensorV2(),
        ]
    )

    return test_transforms