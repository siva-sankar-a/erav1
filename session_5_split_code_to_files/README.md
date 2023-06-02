# Session 5

## <ins>Problem</ins>

- Creating working file structure for training.
- Use the created file structure to train MNIST dataset on Goolge Colab.
- Collect results and prepare documentation for results.

## <ins> Files added </ins>

- ### `utils.py`

    This file contains all the utility functions reqired to train and test a model

    -  ### <ins> `Functions, classes and code snipppets:` </ins>

    > `get_device` 
    - This function checks if the model can be trained on `CUDA` or on `CPU` and returns the current device.
        ``` 
        device = get_device()
        print(device)
        ```
    > `get_train_transforms`
     - This function loads the train transforms
        ``` 
        train_transforms = get_train_transforms()
        ```
    > `get_test_transforms`
     - This function loads the test transforms
        ```
        test_transforms = get_test_transforms()
        ```
    > `get_train_dataset`
     - This function loads the train dataset
        ```
        train_transforms = get_train_transforms()
        train_dataset = get_train_dataset(train_transforms)
        ```
    > `get_test_dataset`
     - This function loads the test dataset
        ```
        test_transforms = get_test_transforms()
        test_dataset = get_test_dataset(test_transforms)
        ```
    > `get_train_dataloader`
     - This function loads train dataloader
        ```
        train_transforms = get_train_transforms()
        train_dataset = get_train_dataset(train_transforms)
        train_dataloader = get_train_dataloader(train_dataset, **kwargs)
        ```
    > `get_test_dataloader`
     - This function loads test dataloader
        ```
        test_transforms = get_test_transforms()
        test_dataset = get_test_dataset(test_transforms)
        test_dataloader = get_test_dataloader(test_dataset, **kwargs)
        ```
    
    > `show_image_grid`
     - This function displays images from a data loader 
        ```
        dataloader = get_test_dataloader(test_dataset, **kwargs)
        ...
        show_image_grid(dataloader)
        ```
        ![Sample output](imagegrid.png)

    > `show_summary`
     - This function displays the summary of a model structure
        ```
        device = ...
        model = Net().to('cpu')
        ...
        show_summary(model, -1, device.type)
        ```
    
    > `train`
     - This function trains the provided `model` on the `train_loader`
        ```
        metrics = { 'train_acc': [], 'train_losses': [],
                        'test_acc': [], 'test_losses': [] }
        ...
        model = ...
        device = ...
        train(model, device, train_dataloader, metrics)
        ``` 

    > `test`
     - This function trains the provided `model` on the `test_loader`
        ```
        metrics = { 'train_acc': [], 'train_losses': [],
                        'test_acc': [], 'test_losses': [] }
        ...
        model = ...
        device = ...
        test(model, device, train_dataloader, metrics)
        ``` 
    
- ### `model.py`
    This file contains the model implemeted for the session. 

    -  ### <ins> `Functions, classes and code snipppets:` </ins>

       > `class Net`

        This class inherits form `nn.Module` and implements 
        the neural network architecture mentioned in summary
        ```
            device = ...
            model = Net().to('cpu')
        ```
       The model architecture is similar to `AlexNet` but with just 4 covolutional blocks and 
       two fully connected layers in the end.
       
       ![Alexnet](alexnet.png)

    ```
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 26, 26]             288
                Conv2d-2           [-1, 64, 24, 24]          18,432
                Conv2d-3          [-1, 128, 10, 10]          73,728
                Conv2d-4            [-1, 256, 8, 8]         294,912
                Linear-5                   [-1, 50]         204,800
                Linear-6                   [-1, 10]             500
    ================================================================
    Total params: 592,660
    Trainable params: 592,660
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.67
    Params size (MB): 2.26
    Estimated Total Size (MB): 2.93
    ----------------------------------------------------------------
    ```

    > `run`

    This function colates all the required components from `utils.py` and `model.py`
    in order to verify if all components can work together in the local environment

    A similar structure is implemented in the Coogle colab notebook in order to train and test the model.
    ```
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
    ```



- `S5.ipynb`

