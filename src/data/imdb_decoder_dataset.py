import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from functools import partial

import torch
import torch.utils.data as data
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import PadTransform, ToTensor, Truncate
from torchtext.datasets import IMDB

from config.env import CHECKPOINT_PATH
from data.imdb_dataset import IMDBDataset, yield_tokens
from data.util import build_vocab, decorated_text_to_ids, shift_tokens_right


class IMDBDecoderDataset(data.Dataset):

    def __init__(self, split="main", max_seq_len=512):

        self.tokenizer = get_tokenizer("basic_english")   

        root_dir = os.path.join(CHECKPOINT_PATH, "IMDBDecoderTask") 
        self.vocab = build_vocab(partial(yield_tokens, self.tokenizer), root_dir + "/vocab_prebuild.pth")

        
        self.truncate = Truncate(max_seq_len - 2)
        self.pad = PadTransform(max_length=max_seq_len, pad_value=self.vocab['<pad>'])
        self.to_tensor = ToTensor()

        inputs =torch.stack([decorated_text_to_ids(text, self.truncate, self.pad, self.tokenizer, self.vocab) 
                             for label, text in IMDB(split=split)])        
        targets = shift_tokens_right(inputs, self.vocab['<pad>'])

        self.data = list(zip(inputs, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def prepare_data_loader():
    batch_size = 32
    train_loader = data.DataLoader(IMDBDecoderDataset(split="train", max_seq_len=24), batch_size=batch_size)
    val_loader   = data.DataLoader(IMDBDecoderDataset(split="test", max_seq_len=24), batch_size=batch_size)

    return train_loader, val_loader, val_loader


if __name__ == "__main__":

    data_set = IMDBDecoderDataset()
    print(data_set[0])

    train_loader, val_loader, _ = prepare_data_loader()
    print(train_loader.dataset[0])
