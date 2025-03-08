import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from functools import partial

import torch
import torch.utils.data as data
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer

from config.env import CHECKPOINT_PATH
from data.util import build_vocab, collate_batch


torchtext.disable_torchtext_deprecation_warning()

def yield_tokens(tokenizer):

    for label, text in IMDB(split="train"):
        yield tokenizer(text)

    for label, text in IMDB(split="test"):
        yield tokenizer(text)


# IMPORTANT: make sure training_loader and val_loader use the same vocab.
class IMDBDataset(data.Dataset):
    def __init__(self, split='train', seq_len=512, size=25000):
        # seq_len: eg. the first 512 words used for prediction
        # size: the max reviews to be used for training

        super().__init__()
        self.size = size
        self.seq_len = seq_len

        self.data_iter = IMDB(split=split)
        
        self.tokenizer = get_tokenizer("basic_english")
        
        root_dir = os.path.join(CHECKPOINT_PATH, "IMDBReviewTask")
        os.makedirs(root_dir, exist_ok=True)
        self.vocab = build_vocab(partial(yield_tokens, self.tokenizer), root_dir + "/vocab_prebuild.pth")

        self.num_categories = len(self.vocab)

        # Convert data to list after processing
        # Torchtext IMDB labels are 1, 2.
        # Must convert to 0, 1 for binary cross entropy, eg. label - 1
        self.data = [
            # For label to be used in binary classification, need to be float type
            (self.text_to_tensor(text), torch.tensor([label-1], dtype=torch.float)) 
            for idx, (label, text) in enumerate(self.data_iter)
        ]

    def text_to_tensor(self, text):
        # only use the first seq_len of words for prediction
        return torch.tensor([self.vocab[token] for idx, token in enumerate(self.tokenizer(text))
                             if idx < self.seq_len], dtype=torch.long)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def prepare_data_loader():
    batch_size = 128
    train_loader = data.DataLoader(IMDBDataset(split='train'), batch_size=batch_size, shuffle=True, collate_fn=collate_batch,
                                   drop_last=True, pin_memory=True,)
    val_loader   = data.DataLoader(IMDBDataset(split='test'), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader  = data.DataLoader(IMDBDataset(split='test'), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    return train_loader,val_loader, test_loader


if __name__ == "__main__":

    batch_size = 64
    train_loader = data.DataLoader(IMDBDataset(split='train'), 
                                               batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                                               collate_fn=collate_batch)
    val_loader = data.DataLoader(IMDBDataset(split='test'), batch_size=batch_size)

    inp_data, label = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", label)

    # check label distribution: 0, 1 are both 50%.
    label_1s = []
    for text, label in train_loader:
        label_1s.append( (label==0).sum()/len(label))

    import numpy as np
    print("Label 1 count", np.mean(label_1s))
