def run():

    import utils as U
    import datasets as ds
    import transforms as tf
    import torch.optim as optim
    from model_mnist_50k_params_CIFAR10 import Net

    device = U.get_device()
    print(device)

    batch_size = 512
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_transforms = tf.get_train_transforms()
    test_transforms = tf.get_test_transforms()

    train_dataset = ds.get_train_dataset(train_transforms)
    test_dataset = ds.get_test_dataset(test_transforms)

    train_dataloader = ds.get_train_dataloader(train_dataset, **kwargs)
    test_dataloader = ds.get_test_dataloader(test_dataset, **kwargs)

    U.show_image_grid(train_dataloader)

    model = Net().to('cpu')

    U.show_summary(model, -1, device.type)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    num_epochs = 20

    metrics = { 'train_acc': [], 'train_losses': [],
                'test_acc': [], 'test_losses': [] }

    labels = {0: 'airplane', 
            1: 'automobile', 
            2: 'bird', 
            3: 'cat', 
            4: 'deer', 
            5: 'dog', 
            6: 'frog', 
            7: 'horse', 
            8: 'ship', 
            9: 'truck'}

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        U.train(model, device, train_dataloader, optimizer, metrics)
        U.test(model, device, test_dataloader, metrics, labels)
        scheduler.step()

    U.display_results(metrics)


if __name__ == '__main__':
    run()