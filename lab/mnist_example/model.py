import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


from dataclasses import dataclass

import pytorch_lightning as pl
from omegaconf import OmegaConf, MISSING


class MNISTModel(pl.LightningModule):
    @dataclass
    class Config:
        drop_prob: float = 0.2
        in_features: int = 28 * 28
        learning_rate: float = 0.001 * 8
        out_features: int = 10
        hidden_dim: int = 1000

        optimizer_name: str = "adam"

        data_root: str = "./datasets"
        epochs: int = 20
        batch_size: int = 2

    def __init__(self, hparams: Config, *args, **kwargs):
        super().__init__()

        self.hparams = hparams

        self.c_d1 = nn.Linear(in_features=self.hparams.in_features, out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim, out_features=self.hparams.out_features)

        self.example_input_array = torch.zeros(2, 1, 28, 28)

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.c_d1(x.view(x.size(0), -1))
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {"val_loss": val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {"test_loss": test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
        tensorboard_logs = {"test_loss": avg_loss, "test_acc": test_acc}
        return {"test_loss": avg_loss, "log": tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        MNIST(self.hparams.data_root, train=True, download=True, transform=transforms.ToTensor())
        MNIST(self.hparams.data_root, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        self.mnist_train = MNIST(self.hparams.data_root, train=True, download=False, transform=transform)
        self.mnist_test = MNIST(self.hparams.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=4)
