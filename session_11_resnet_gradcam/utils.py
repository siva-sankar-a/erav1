import torch
from torchvision import datasets, transforms
from torchsummary import summary
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report

import numpy as np

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

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
                                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)) 
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

def show_image_grid(data_loader, n_channels=3):
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
        if n_channels > 1:
            img = (batch_data[i].squeeze(0) * 255).int()
            plt.imshow(img.permute(1, 2, 0), cmap='jet')
        else:
            plt.imshow(batch_data[i].squeeze(0), cmap='grey')

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
    summary(model, input_size=(3, 32, 32), batch_size=batch_size, device=device)

def _get_correct_pred_count(prediction, labels):
  return prediction.argmax(dim=1).eq(labels).sum().item()

def train(model, device, train_loader, optimizer, metrics, scheduler=None):

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

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = cross_entropy_loss(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        correct += _get_correct_pred_count(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    metrics['train_acc'].append(100 * correct / processed)
    metrics['train_losses'].append(train_loss / len(train_loader))

    return metrics

def test(model, device, test_loader, metrics, label_map=None, get_misclassified=True):

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

    pred = []
    actual = []
    misclassified_data = []

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += cross_entropy_loss(output, target).item()  # sum up batch loss

            correct += _get_correct_pred_count(output, target)

            pred_batch = output.argmax(dim=1).cpu().tolist()
            actual_batch = target.cpu().tolist()

            pred += pred_batch
            actual += actual_batch

            if get_misclassified:
                for i in range(data.shape[0]):
                    if pred_batch[i] != actual_batch[i]:
                        _misclassified_data = {
                            'pred': pred_batch[i],
                            'actual': actual_batch[i],
                            'data': data[i].detach().cpu().numpy()
                        }
                        misclassified_data.append(_misclassified_data)

    if label_map:
        pred = [label_map[p] for p in pred]
        actual = [label_map[a] for a in actual]

    print(classification_report(actual, pred))


    test_loss /= len(test_loader.dataset)

    metrics['test_acc'].append(100. * correct / len(test_loader.dataset))
    metrics['test_losses'].append(test_loss)
    metrics['misclassified_data'] = misclassified_data

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

def gradcam(df, model, target_layers, labels):
    _misclassified_batch = np.array(df['data'].tolist())

    # target_layers = [model.layer3[-1]]
    input_tensor = torch.from_numpy(_misclassified_batch)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(t) for t in df['actual']]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    mean_R, mean_G, mean_B = (0.4913997551666284, 0.48215855929893703, 0.446530913373161)
    std_R, std_G, std_B = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

    fig = plt.figure(figsize=(24, 10))

    for i, row in df.reset_index(drop=True).iterrows():
        ax = plt.subplot(4, 10, (2 * i)+2)
        plt.axis('off')

        cam_result = grayscale_cam[i, :]

        _misclassified_batch[i, 0, :, :] = (_misclassified_batch[i, 0, :, :] * std_R) + mean_R
        _misclassified_batch[i, 1, :, :] = (_misclassified_batch[i, 1, :, :] * std_G) + mean_G
        _misclassified_batch[i, 2, :, :] = (_misclassified_batch[i, 2, :, :] * std_B) + mean_B

        visualization = show_cam_on_image(_misclassified_batch[i].transpose((1, 2, 0)), cam_result, use_rgb=True, image_weight=0.6)

        plt.imshow(visualization, cmap='jet')

        ax = plt.subplot(4, 10, (2 * i)+1)
        plt.axis('off')
        plt.imshow(_misclassified_batch[i].transpose((1, 2, 0)), cmap='jet')

        ax.set_title(f"actual: {labels[row['actual']]} \n predicted: {labels[row['pred']]}")

    plt.show()
    plt.savefig('gradcam.jpg')