import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from utils.types_ import *


# Device configuration
# GPU_NUM = 0
DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Experiment(pl.LightningModule):
    def __init__(self, model, params):
        super(Experiment, self).__init__()

        self.model = model
        self.params = params
        self._loss = nn.BCELoss()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, docs, doc_lens):
        return self.model(docs, doc_lens)

    def loss_function(self, preds, labels):
        bce_loss = self._loss(preds, labels)
        return bce_loss

    def accuracy(self, preds, labels):
        preds = torch.round(preds)
        corrects = (preds == labels).float().sum()
        acc = corrects / labels.numel()
        return acc

    def training_step(self, batch, batch_idx):
        features, targets, doc_lens, _ = batch

        preds = self.forward(features, doc_lens)
        train_loss = self.loss_function(preds, targets)
        train_acc = self.accuracy(preds, targets)
        log_dict = {"train_acc": train_acc, "train_loss": train_loss}

        output = OrderedDict(
            {
                "loss": train_loss,
                "progress_bar": {"train_acc": train_acc},
                "log": log_dict,
            }
        )
        return output

    def validation_step(self, batch, batch_idx):
        features, targets, doc_lens, _ = batch

        preds = self.forward(features, doc_lens)
        val_loss = self.loss_function(preds, targets)
        val_acc = self.accuracy(preds, targets)

        tqdm_dict = {"val_acc": val_acc, "val_loss": val_loss}
        output = OrderedDict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "log": tqdm_dict,
                "progress_bar": tqdm_dict,
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": val_loss_mean, "val_acc": val_acc_mean}

    #     def test_step(self, batch, batch_idx):
    #         sequences, labels, keywords = batch

    #         preds = self.forward(sequences)
    #         test_loss = self.loss_function(preds, labels)
    #         return {"test_loss": test_loss}

    #     def test_epoc_end(self, outputs):
    #         val_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
    #         return {"test_loss": val_loss_mean}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params["LR"], weight_decay=1e-5)
