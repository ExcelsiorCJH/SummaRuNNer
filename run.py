import argparse
import dill
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger  # pip install test-tube

from functools import partial
from collections import OrderedDict

from experiment import Experiment
from model import SummaRunner
from utils.data import SumDataset, Feature
from utils.preprocess import build_vocab, collate_fn
from utils.types_ import *

import warnings

warnings.filterwarnings(action="ignore")


parser = argparse.ArgumentParser(description="Generic runner for BiLSTMAttn models")

parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
    default="./config.yaml",
)

args = parser.parse_args()

with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# ----------------
# DataLoader
# ----------------

# data path
train_path = config["exp_params"]["train_path"]
valid_path = config["exp_params"]["valid_path"]
vocab_path = config["exp_params"]["vocab_path"]

# vocab
with open(vocab_path, "rb") as f:
    word2id = dill.load(f)

# pretrained vectors

# Feature class
feature = Feature(word2id)

# Dataset
trainset = SumDataset(train_path)
validset = SumDataset(valid_path)

# DataLoader
train_loader = DataLoader(
    dataset=trainset,
    batch_size=config["exp_params"]["batch_size"],
    shuffle=True,
    collate_fn=partial(collate_fn, feature=feature),
    num_workers=8,
)

valid_loader = DataLoader(
    dataset=validset,
    batch_size=config["exp_params"]["batch_size"],
    shuffle=False,
    collate_fn=partial(collate_fn, feature=feature),
    num_workers=8,
)

# SetUp Model
# ----------------

# vocab_size
config["model_params"]["vocab_size"] = len(word2id)
# num_class
config["model_params"]["num_class"] = 1

model = SummaRunner(**config["model_params"])
experiment = Experiment(model, config["exp_params"])

# ----------------
# TestTubeLogger
# ----------------
tt_logger = TestTubeLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["logging_params"]["name"],
    debug=False,
    create_git_tag=False,
)

# ----------------
# Checkpoint
# ----------------
checkpoint_callback = ModelCheckpoint(
    filepath="./checkpoints/summarunner{epoch:02d}_{val_loss:.3f}",
    monitor="val_loss",
    verbose=True,
    save_top_k=5,
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=True)

# ----------------
# Trainer
# ----------------

runner = Trainer(
    default_save_path=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    log_save_interval=100,
    train_percent_check=1.0,
    val_percent_check=1.0,
    num_sanity_val_steps=5,
    early_stop_callback=early_stopping,
    checkpoint_callback=checkpoint_callback,
    **config["trainer_params"],
)

# ----------------
# Start Train
# ----------------
runner.fit(experiment, train_loader, valid_loader)