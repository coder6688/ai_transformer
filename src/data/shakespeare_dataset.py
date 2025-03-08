import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from functools import partial

import torch
import torch.utils.data as data
import torchtext
from torchtext.datasets import IMDB
from torchtext.transforms import PadTransform, ToTensor, Truncate

from config.env import CHECKPOINT_PATH
from data.util import build_vocab, collate_batch, decorated_text_to_ids, shakespeare_tokenizer, shift_tokens_right

torchtext.disable_torchtext_deprecation_warning()


def yield_tokens(tokenizer):
    file_path = Path(__file__).parent.parent.parent / "shakespeare" / "alllines.txt"
    print(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line and len(cleaned_line) > 2:
                yield tokenizer(cleaned_line[1:-1]) # remove the beginning/ending quote 

def yield_line():
    file_path = Path(__file__).parent.parent.parent / "shakespeare" / "alllines.txt"
    print(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line and len(cleaned_line) > 2:
                yield cleaned_line[1:-1] # remove the beginning/ending quote     



class ShakespeareDecoderDataset(data.Dataset):

    def __init__(self, split="train", max_seq_len=80, max_size=None):
        # ignore split for now

        self.tokenizer = shakespeare_tokenizer 

        root_dir = os.path.join(CHECKPOINT_PATH, "ShakespeareDecoderTask") 
        self.vocab = build_vocab(partial(yield_tokens, self.tokenizer), root_dir + "/vocab_prebuild.pth")

        self.truncate = Truncate(max_seq_len - 2)

        inputs, targets = [], []
        for idx, text in enumerate(yield_line()):
            if max_size is not None and idx >= max_size:
                break
            input = decorated_text_to_ids(text, truncate=self.truncate, pad=None, tokenizer=self.tokenizer, vocab=self.vocab)
            target = torch.cat([input.clone()[1:], torch.tensor([self.vocab['<eos>']])])
            

            # target = input.clone()[1:]
            # input = input[0:-1]
            
            inputs.append(input)
            targets.append(target)

        self.data = list(zip(inputs, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def prepare_data_loader():
    batch_size = 64
    data_set = ShakespeareDecoderDataset(split="test", max_seq_len=80)

    train_loader = data.DataLoader(data_set, batch_size=batch_size, collate_fn=partial(collate_batch, pad_id=data_set.vocab['<pad>']))
    val_loader   = data.DataLoader(data_set, batch_size=batch_size, collate_fn=partial(collate_batch, pad_id=data_set.vocab['<pad>']))

    return train_loader, val_loader, val_loader


if __name__ == "__main__":

    data_set = ShakespeareDecoderDataset()
    print(data_set[0:3])

    train_loader, val_loader, _ = prepare_data_loader()
    print(train_loader.dataset[2:3])

    # for input, label in train_loader:
    #     print(input)
    #     print(label)