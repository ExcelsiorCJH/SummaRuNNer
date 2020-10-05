import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.types_ import *


# Device configuration
DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


class SentenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
        pretrained_vectors: np.ndarray = None,
    ):
        super().__init__()

        self.vocab_size = (vocab_size,)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directs = 1
        if bidirectional:
            self.num_directs = 2

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_vectors is not None:
            self.embed.weight.data.copy_(pretrained_vectors)
        else:
            nn.init.xavier_uniform_(self.embed.weight)

        self.bilstm = nn.LSTM(
            self.embed_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            # dropout=dropout,
        )
        # self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def avg_pool1d(self, sequences, seq_lens):
        out = []
        for idx, tensor in enumerate(sequences):
            tensor = tensor[: seq_lens[idx], :]
            tensor = torch.t(tensor).unsqueeze(0)
            out.append(F.avg_pool1d(tensor, tensor.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, docs):
        sent_lens = torch.sum(torch.sign(docs), dim=1).data

        x = self.embed(docs)
        output, _ = self.bilstm(x)
        output = self.avg_pool1d(output, sent_lens)
        # output = self.linear(output)

        return output


class DocumentEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directs = 1
        if bidirectional:
            self.num_directs = 2

        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # dropout=dropout,
        )
        # self.linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def pad_doc(self, sents, doc_lens):
        pad_dim = sents.size(1)
        max_doc_len = max(doc_lens)
        sent_input = []
        start = 0
        for doc_len in doc_lens:
            stop = start + doc_len
            valid = sents[start:stop]
            start = stop
            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0))
            else:
                pad = Variable(torch.zeros(max_doc_len - doc_len, pad_dim)).to(DEVICE)
                sent_input.append(torch.cat([valid, pad]).unsqueeze(0))

        sent_input = torch.cat(sent_input, dim=0)
        return sent_input

    def avg_pool1d(self, sequences, seq_lens):
        out = []
        for idx, tensor in enumerate(sequences):
            tensor = tensor[: seq_lens[idx], :]
            tensor = torch.t(tensor).unsqueeze(0)
            out.append(F.avg_pool1d(tensor, tensor.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, sents, doc_lens):
        # make sent features(pad with zeros)
        x = self.pad_doc(sents, doc_lens)
        output, hidden = self.bilstm(x)
        # output = self.linear(output)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
        pretrained_vectors: np.ndarray = None,
    ):
        super().__init__()

        self.sent_encoder = SentenceEncoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            dropout_p=dropout_p,
            pretrained_vectors=pretrained_vectors,
        )

        self.doc_encoder = DocumentEncoder(
            2 * hidden_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            dropout_p=dropout_p,
        )

    def forward(self, docs, doc_lens):
        encoded_sents = self.sent_encoder(docs)
        encoded_docs = self.doc_encoder(encoded_sents, doc_lens)
        return encoded_docs
