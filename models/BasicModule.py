import torch
import torch.nn as nn
from torch.autograd import Variable


class BasicModule(nn.Module):

    def __init__(self, args):
        super(BasicModule, self).__init__()
        self.agrs = args
        self.model_name = str(type(self))

    def pad_doc(self, words_out, doc_lens):
        pad_dim = words_out.size(1)
        max_doc_len = max(doc_lens)
        sent_input = []
        start = 0
        for doc_len in doc_lens:
            stop = start + doc_len
            valid = words_out[start:stop]
            start = stop
            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0))
            else:
                pad = Variable(torch.zeros(max_doc_len - doc_len, pad_dim))
                if self.args.device:
                    pad = pad.cuda()
                sent_input.append(torch.cat([valid, pad]).unsqueeze(0))
        
        sent_input = torch.cat(sent_input, dim=0)
        return sent_input

    def save(self):
        checkpoint = {'model':self.state_dict(), 'args': self.args}
        best_path = '%s%s_seed_%d.pt' % (self.args.save_dir,self.model_name,self.args.seed)
        torch.save(checkpoint,best_path)

        return best_path
    
    def load(self, best_path):
        if self.args.device:
            data = torch.load(best_path)['model']
        else:
            data = torch.load(best_path, 
                              map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device:
            return self.cuda()
        else:
            return self
