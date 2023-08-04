
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch.optim as optim
import utils as U
import datasets as ds
import transforms as tf

from model.model_david_net_90_acc import Net
# from model.model_50k_params_CIFAR10_bn_70_acc import Net
from pytorch_lightning.callbacks import Callback

class NetPL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Net()
        self.num_classes = 10
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):

        num_epochs = 20
        momentum = 0.9
        max_lr = 1E-02
        regularization = None
        epochs_up = 7
        base_momentum = 0.85
        div_factor = 100

        steps_per_epoch = self.trainer.estimated_stepping_batches
        total_steps = num_epochs * steps_per_epoch
        pct_start = epochs_up / num_epochs

        optimizer = optim.Adam(self.model.parameters(), lr=0.1)
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

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def training_step(self, train_batch, batch_idx):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        data, target = train_batch

        pred = self.model(data)

        self.accuracy(pred, target)
        loss = cross_entropy_loss(pred, target)

        self.log('train_acc_step', self.accuracy, prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        data, target = val_batch

        pred = self.model(data)
        loss = cross_entropy_loss(pred, target)

        self.accuracy(pred, target)
        self.log('val_acc_step', self.accuracy, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)

class MisclassifiedCollector(Callback):

    def __init__(self):
        super().__init__()
        self.misclassified_data = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        data, target = batch

        pred_batch = outputs.argmax(dim=1).cpu().tolist()
        actual_batch = target.cpu().tolist()
        self.misclassified_data = []

        for i in range(data.shape[0]):
            if pred_batch[i] != actual_batch[i]:
                _misclassified_data = {
                    'pred': pred_batch[i],
                    'actual': actual_batch[i],
                    'data': data[i].detach().cpu().numpy()
                }
                self.misclassified_data.append(_misclassified_data)

def run():
    # data
    batch_size = 512
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}


    train_transforms = tf.get_CIFAR10_train_transforms()
    test_transforms = tf.get_CIFAR10_test_transforms()

    train_dataset = ds.get_CIFAR10_albumentations_train_dataset(train_transforms)
    test_dataset = ds.get_CIFAR10_albumentations_test_dataset(test_transforms)

    train_dataloader = ds.get_train_dataloader(train_dataset, **kwargs)
    test_dataloader = ds.get_test_dataloader(test_dataset, **kwargs)

    # model
    model = NetPL()

    # training
    logger = TensorBoardLogger("tb_logs", name="davidnet")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    misclassified_collector = MisclassifiedCollector()

    trainer = pl.Trainer(precision=16, logger=logger, max_epochs=20, callbacks=[lr_monitor, misclassified_collector])
    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    run()