import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .BasicModule import BasicModule


class RNN_RNN(BasicModule):
    
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = "RNN_RNN"
        self.args = args
        
        vocab_size = args.embed_num
        embed_dim = args.embed_dim
        hidden_size = args.hidden_size
        seg_num = args.seg_num
        pos_num = args.pos_num
        pos_dim = args.pos_dim

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.abs_pos_embed = nn.Embedding(pos_num, pos_dim)  # absolute postion
        self.rel_pos_embed = nn.Embedding(seg_num, pos_dim)  # relative position
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(input_size=embed_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                bidirectional=True)

        self.sent_RNN = nn.GRU(input_size=2*hidden_size,
                                hidden_size=hidden_size,
                                batch_first=True,
                                bidirectional=True)

        self.fc = nn.Linear(2*hidden_size, 2*hidden_size)

        # Parameters of Classification Layer
        # P(y_j = 1|h_j, s_j, d), Eq.6 in SummaRuNNer paper 
        self.content = nn.Linear(2*hidden_size, 1, bias=False)
        self.salience = nn.Bilinear(2*hidden_size, 2*hidden_size, 1, bias=False)
        self.novelty = nn.Bilinear(2*hidden_size, 2*hidden_size, 1, bias=False)
        self.abs_pos = nn.Linear(pos_dim, 1, bias=False)
        self.rel_pos = nn.Linear(pos_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    
    def max_pool1d(self, x, seq_lens):
        out = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t, t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out
    
    def avg_pool1d(self, x, seq_lens):
        out = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t, t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out


    def forward(self, x, doc_lens):
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embed(x)

        # word-level GRU
        hidden_size = self.args.hidden_size
        x = self.word_RNN(x)[0]
        word_out = self.avg_pool1d(x, sent_lens)
        # word_out = self.max_pool1d(x, sent_lens)
        
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out, doc_lens)

        # sent-level GRU
        sent_out = self.sent_RNN(x)[0]
        docs = self.avg_pool1d(sent_out, doc_lens)
        # docs = self.max_pool1d(sent_out, doc_lens)

        probs = []
        for index, doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index, :doc_len, :]
            doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1, 2*hidden_size))
            if self.args.device:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                rel_index =int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device:
                    rel_index = rel_index.cuda()
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