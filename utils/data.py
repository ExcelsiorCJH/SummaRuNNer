import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords

from .types_ import *


class SumDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]["doc"]
        labels = self.data[idx]["labels"]
        summaries = self.data[idx]["summaries"]

        return doc, labels, summaries


class Feature:
    def __init__(self, word2id):
        self.word2id = word2id
        self.id2word = {idx: word for word, idx in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"

    def __len__(self):
        return len(self.word2id)

    def i2w(self, idx):
        return self.id2word[idx]

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    ###################
    # Create Features #
    ###################
    def make_features(
        self,
        docs,
        labels_list,
        summaries_list,
        sent_trunc=128,
        doc_trunc=100,
        split_token="\n",
    ):

        # trunc document
        # 문서 내 doc_trunc 문장 개수까지 가져옴
        sents_list, targets, doc_lens, summaries = [], [], [], []
        for doc, labels, summary in zip(docs, labels_list, summaries_list):
            sents = doc.split(split_token)
            labels = labels.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            oracle = [sent for sent, label in zip(sents, labels) if label == 1]

            sents_list.extend(sents)
            targets.extend(labels)
            doc_lens.append(len(sents))
            summaries.append(summary)

        # trunc or pad sent
        # 문장 내 sent_trunc 단어 개수까지 가져옴
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.PAD_IDX for _ in range(max_sent_len - len(sent))] + [
                self.w2i(w) for w in sent
            ]
            features.append(feature)

        return features, targets, doc_lens, summaries

    def make_predict_features(
        docs,
        sent_trunc=128,
        doc_trunc=100,
        split_token=". ",
    ):

        sents_list, doc_lens = [], []
        for doc in docs:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            sents_list.extend(sents)
            doc_lens.append(len(sents))

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.PAD_IDX for _ in range(max_sent_len - len(sent))] + [
                self.w2i(w) for w in sent
            ]
            features.append(feature)

        return features, doc_lens