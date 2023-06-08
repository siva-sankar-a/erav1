import torch
from torchvision import datasets, transforms
from torchsummary import summary
import torch.nn.functional as F
from tqdm import tqdm

def get_device():
    '''
    This function checks if the model can be trained on `CUDA` or on `CPU`
    '''
    # CUDA?
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA Available?", cuda)
    return device

def get_train_transforms():
    '''
    This function loads the train transforms

    Usage:
    >>> train_transforms = get_train_transforms()
    '''
    # Train data transformations
    train_transforms = transforms.Compose([
    # transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    # transforms.Resize((28, 28)),
    # transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
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
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    return test_transforms

def get_train_dataset(train_transforms):
    '''
    This function loads the train dataset

    Usage:
    >>> train_transforms = get_train_transforms()
    >>> train_dataset = get_train_dataset(train_transforms)
    '''
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    return train_data

def get_test_dataset(test_transforms):
    '''
    This function loads the test dataset

    Usage:
    >>> test_transforms = get_test_transforms()
    >>> test_dataset = get_test_dataset(test_transforms)
    '''
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
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

def show_image_grid(data_loader):
    '''
    This function displays images from a data loader 

    Usage:
    >>> dataloader = get_test_dataloader(test_dataset, **kwargs)
    ...
    >>> show_image_grid(dataloader)
    '''

    import matplotlib.pyplot as plt

    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
    
    plt.show()

def show_summary(model, batch_size=-1, device='cpu'):
    '''
    This function displays the summary of a model structure

    Usage:
    >>> device = ...
    >>> model = Net().to('cpu')
    ...
    >>> show_summary(model, -1, device.type)
    '''
    summary(model, input_size=(1, 28, 28), batch_size=batch_size, device=device)

def _get_correct_pred_count(prediction, labels):
  return prediction.argmax(dim=1).eq(labels).sum().item()

def train(model, device, train_loader, optimizer, metrics):

    '''
    This function trains the provided `model` on the `train_loader`

    Usage 
        >>> metrics = { 'train_acc': [], 'train_losses': [],
                        'test_acc': [], 'test_losses': [] }
        ...
        model = ...
        device = ...
        >>> train(model, device, train_dataloader, metrics)
    ''' 

    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += _get_correct_pred_count(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    metrics['train_acc'].append(100 * correct / processed)
    metrics['train_losses'].append(train_loss / len(train_loader))

    return metrics

def test(model, device, test_loader, metrics):

    '''
    This function tests the provided `model` on the `test_loader`

    Usage 
        >>> metrics = { 'train_acc': [], 'train_losses': [],
                        'test_acc': [], 'test_losses': [] }
        ...
        model = ...
        device = ...
        >>> test(model, device, test_dataloader, metrics)
    ''' 
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += _get_correct_pred_count(output, target)


    test_loss /= len(test_loader.dataset)

    metrics['test_acc'].append(100. * correct / len(test_loader.dataset))
    metrics['test_losses'].append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return metrics

def display_results(metrics):

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(metrics['train_losses'])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(metrics['train_acc'])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(metrics['test_losses'])
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(metrics['test_acc'])
    axs[1, 1].set_title("Test Accuracy")

    plt.show()