import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import pytorch_lightning as pl
import torchmetrics
import torch.optim as optim

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        pct_start = 0.3

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

        output = self.model(data)

        logits = F.log_softmax(output, dim=1)
        pred = torch.argmax(logits, dim=1)

        self.accuracy(pred, target)
        loss = cross_entropy_loss(output, target)

        self.log('train_acc_step', self.accuracy, prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        data, target = val_batch

        output = self.model(data)

        logits = F.log_softmax(output, dim=1)
        pred = torch.argmax(logits, dim=1)

        loss = cross_entropy_loss(output, target)

        self.accuracy(pred, target)
        self.log('val_acc_step', self.accuracy, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return logits