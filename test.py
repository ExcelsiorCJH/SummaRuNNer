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

# test
parser.add_argument('-seed', type=int, default=42)
parser.add_argument('-embedding', type=str, default='./data/embedding.npz')
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-word2id', type=str, default='./data/word2id.json')
parser.add_argument('-load_dir', type=str, default='./checkpoints/RNN_RNN_seed_42.pt')
parser.add_argument('-test_dir', type=str, default='./data/test.json')
parser.add_argument('-ref', type=str, default='./outputs/summarunner/ref')
parser.add_argument('-hyp', type=str, default='./outputs/summarunner/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=15)

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


def test():
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()

    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        features, _, summaries, doc_lens = vocab.make_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id, doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk, doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            hyp = [doc[idx] for idx in topk_indices]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref, str(file_id)+'.txt'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.hyp, str(file_id)+'.txt'), 'w') as f:
                f.write('\n'.join(hyp))
            start = stop
            file_id += 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict(examples):
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    pred_dataset = utils.Dataset(examples)

    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda sotrage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()

    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(pred_iter):
        features, doc_lens = vocab.make_predict_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id, doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk, doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch[doc_id].split('. ')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            with open(os.path.join(args.hyp, str(file_id)+'.txt'), 'w') as f:
                f.write('. '.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


if __name__ == "__main__":
    if args.predict:
        with open(args.filename) as f:
            bod = [f.read()]
        predict(bod)
    else:
        print('Test...')
        test()