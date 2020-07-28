import os
import argparse
import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils

from time import time
from tqdm import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')

# model
parser.add_argument('-save_dir', type=str, default='checkpoints/')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=100)
parser.add_argument('-seg_num', type=int, default=10)
parser.add_argument('-kernel_num', type=int, default=100)  # for CNN_RNN
parser.add_argument('-kernel_sizes', type=str, default='3,4,5')  # for CNN_RNN
parser.add_argument('-model', type=str, default='RNN_RNN')
parser.add_argument('-hidden_size', type=int, default=200)

# train
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-epochs', type=int, default=5)
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-train_dir', type=str, default='./data/train.json')
parser.add_argument('-val_dir', type=str, default='./data/val.json')
parser.add_argument('-embedding', type=str, default='./data/embedding.npz')
parser.add_argument('-word2id', type=str, default='./data/word2id.json')
parser.add_argument('-report_every', type=int, default=500)
parser.add_argument('-seq_trunc', type=int, default=50)
parser.add_argument('-max_norm', type=float, default=1.0)

# device
parser.add_argument('-device', type=int, default=1)
# option
parser.add_argument('-test', action='store_true')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-predict', action='store_true')

args = parser.parse_args()
use_gpu = args.device is not None

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


# accuracy
def accuracy(logits, labels):
    preds = torch.round(logits)
    corrects = (preds == labels).sum().float()
    acc = corrects / labels.numel()
    return acc


def eval(net, vocab, data_loader, criterion):
    net.eval()
    total_loss = 0
    total_acc = 0
    batch_num = 0
    for batch in tqdm(data_loader, desc='Eval', position=1):
        features, targets, _, doc_lens = vocab.make_features(batch)
        features, targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_lens)
        loss = criterion(probs, targets)
        acc = accuracy(probs, targets)
        total_loss += loss.item()
        total_acc += acc
        batch_num += 1
    loss = total_loss / batch_num
    acc = total_acc / batch_num
    net.train()
    return loss, acc


def train():
    logging.info('Loading vocab, train and val dataset...')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]  # for CNN_RNN
    
    # build model
    net = getattr(models, args.model)(args, embed)
    if use_gpu:
        net.cuda()

    # load dataset
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    
    # loss function
    criterion = nn.BCELoss()
    
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))

    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    # Tensorbard
    writer = SummaryWriter(f'runs/{args.model}')

    t1 = time()
    for epoch in tqdm(range(1, args.epochs+1), desc='Epoch', position=0):
        for i, batch in enumerate(tqdm(train_loader, desc='Train', position=1)):
            features, targets, _, doc_lens = vocab.make_features(batch)
            features, targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            
            probs = net(features, doc_lens)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            optimizer.step()

            # TensorBoard
            train_acc = accuracy(probs, targets)
            writer.add_scalar('train_loss_batch', 
                              loss, epoch * len(train_loader) + i)
            writer.add_scalar('train_acc_batch',
                              train_acc, epoch * len(train_loader) + i)

            if args.debug:
                print(f'Batch ID: {i}, Loss: {loss.item()}, Acc: {train_acc}')
                continue
            
            if i % args.report_every == 0:
                cur_loss, cur_acc = eval(net, vocab, valid_loader, criterion)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_path = net.save()
                logging.info(f'Epoch: {epoch}, Min_Val_Loss: {min_loss}, Cur_Val_Loss: {cur_loss}, Cur_Val_Acc: {cur_acc}')

                # TensorBoard
                writer.add_scalar('valid_loss', 
                                  cur_loss, epoch)
                writer.add_scalar('valid_acc', 
                                  cur_acc, epoch)
    
    t2 = time()
    logging.info('Total Time:%f h' %((t2-t1)/3600))


if __name__ == "__main__":
    train()