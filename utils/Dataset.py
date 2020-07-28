import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset, self).__init__()
        # data: {'sents': 'x\nx\nx\nx', 'labels': 'x\nx\nx\nx', 'summaries': 'xx'}
        self.examples = examples
        self.training = False

    def train(self):
        self.training = True
        return self
    
    def test(self):
        self.training = False
        return self
    
    def shuffle(self, words):
        np.random.shuffle(words)
        return ' '.join(words)
    
    def dropout(self, words, p=0.3):
        l = len(words)
        drop_index = np.random.choice(l,int(l*p))
        keep_words = [words[i] for i in range(l) if i not in drop_index]
        return ' '.join(keep_words)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex
        #words = ex['sents'].split()
        #guess = np.random.random()

        #if self.training:
        #    if guess > 0.5:
        #        sents = self.dropout(words,p=0.3)
        #    else:
        #        sents = self.shuffle(words)
        #else:
        #    sents = ex['sents']
        #return {'id':ex['id'],'sents':sents,'labels':ex['labels']}
        
    def __len__(self):
        return len(self.examples)