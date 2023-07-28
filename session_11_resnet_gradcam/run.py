def run():

    import utils as U
    import datasets as ds
    import transforms as tf
    import torch.optim as optim

    from model.resnet import ResNet18

    device = U.get_device()
    print(device)

    batch_size = 512
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_transforms = tf.get_CIFAR10_DavidNet_train_transforms()
    test_transforms = tf.get_CIFAR10_DavidNet_test_transforms()

    train_dataset = ds.get_CIFAR10_albumentations_train_dataset(train_transforms)
    test_dataset = ds.get_CIFAR10_albumentations_test_dataset(test_transforms)

    train_dataloader = ds.get_train_dataloader(train_dataset, **kwargs)
    test_dataloader = ds.get_test_dataloader(test_dataset, **kwargs)

    U.show_image_grid(train_dataloader)

    model = ResNet18().to('cpu')

    U.show_summary(model, -1, device.type)

    num_epochs = 35
    momentum = 0.9
    max_lr = 5.34E-04
    regularization = None
    epochs_up = 7
    base_momentum = 0.85
    div_factor = 100

    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    pct_start = epochs_up / num_epochs

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=max_lr,
                                            total_steps=total_steps,
                                            epochs=num_epochs,
                                            steps_per_epoch=steps_per_epoch,
                                            pct_start=pct_start,
                                            anneal_strategy='linear',
                                            cycle_momentum=True,
                                            base_momentum=base_momentum,
                                            max_momentum=momentum,
                                            div_factor=div_factor,
                                            verbose=False)


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
        U.train(model, device, train_dataloader, optimizer, metrics, scheduler)
        U.test(model, device, test_dataloader, metrics, labels)
        # scheduler.step()
        break

    U.display_results(metrics)


if __name__ == '__main__':
    run()