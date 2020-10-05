import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .encoder import SentenceEncoder, DocumentEncoder
from .encoder import Encoder


# Device configuration
DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


class SummaRunner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_class: int = 1,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        pos_dim: int = 50,
        pos_num: int = 100,
        seg_num: int = 10,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
        maxlen: int = 50,
        pretrained_vectors: np.ndarray = None,
    ):
        super(SummaRunner, self).__init__()

        self.hidden_dim = hidden_dim
        self.abs_pos_embed = nn.Embedding(pos_num, pos_dim)  # absolute postion
        self.rel_pos_embed = nn.Embedding(seg_num, pos_dim)  # relative position

        self.encoder = Encoder(
            vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout_p
        )

        self.fc = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # Parameters of Classification Layer
        # P(y_j = 1|h_j, s_j, d), Eq.6 in SummaRuNNer paper
        self.content = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.salience = nn.Bilinear(2 * hidden_dim, 2 * hidden_dim, 1, bias=False)
        self.novelty = nn.Bilinear(2 * hidden_dim, 2 * hidden_dim, 1, bias=False)
        self.abs_pos = nn.Linear(pos_dim, 1, bias=False)
        self.rel_pos = nn.Linear(pos_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    def avg_pool1d(self, sequences, seq_lens):
        out = []
        for idx, tensor in enumerate(sequences):
            tensor = tensor[: seq_lens[idx], :]
            tensor = torch.t(tensor).unsqueeze(0)
            out.append(F.avg_pool1d(tensor, tensor.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, docs, doc_lens):
        sent_out = self.encoder(docs, doc_lens)
        docs = self.avg_pool1d(sent_out, doc_lens)

        probs = []
        for index, doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index, :doc_len, :]
            doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1, 2 * self.hidden_dim)).to(DEVICE)
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]])).to(DEVICE)
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]])).to(DEVICE)
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)

                # classification layer
                content = self.content(h)
                salience = self.salience(h, doc)
                novelty = -1 * self.novelty(h, torch.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                # P(y_j = 1|h_j, s_j, d) Eq.6 in SummaRuNNer paper
                prob = torch.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob, h)
                probs.append(prob)

        return torch.cat(probs).squeeze()