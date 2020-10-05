import torch

from collections import Counter, defaultdict
from nltk.corpus import stopwords
from tqdm import tqdm
from .types_ import *


def build_vocab(
    dataset: JSONType, stopwords: List[str] = stopwords.words("english"), num_words: int = 40000
):
    # 1. tokenization
    all_tokens = []
    for data in tqdm(dataset):
        sents = data["doc"].split("\n")
        for sent in sents:
            tokens = sent.lower().split(" ")
            all_tokens.extend([token for token in tokens if token not in stopwords])

    # 2. build vocab
    vocab = Counter(all_tokens)
    vocab = vocab.most_common(num_words)

    # 3. add pad & unk tokens
    word_index = defaultdict()
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    for idx, (word, _) in enumerate(vocab, 2):
        word_index[word] = idx

    index_word = {idx: word for word, idx in word_index.items()}

    return word_index, index_word


def collate_fn(batch, feature, doc_trunc=100):
    docs = [entry[0] for entry in batch]
    labels_list = [entry[1] for entry in batch]
    summaries = [entry[2] for entry in batch]

    features, targets, doc_lens, summaries = feature.make_features(
        docs, labels_list, summaries, doc_trunc=doc_trunc
    )
    
    features = torch.LongTensor(features)
    targets = torch.FloatTensor(targets)
    return features, targets, doc_lens, summaries
