import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from functools import partial

import torch
import torch.utils.data as data
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import PadTransform, ToTensor

from config.env import CHECKPOINT_PATH
from data.imdb_dataset import yield_tokens
from data.util import build_vocab, decorated_text_to_ids, shift_tokens_right

texts = [
    "Hello, who are you?",
    "I love machine learning",
    "Transformers are awesome",
    ]

# test_texts = [
#     "Hello, how are you?",
#     "I love machine learning",
#     "Transformers are awesome",
#     ]

test_texts = [
    "What is up?",
    "Who are you?",
    "why is that?"
    ]

class SampleDecoderDataset(data.Dataset):

    def __init__(self, texts=texts, max_seq_len=8, max_size=None):

        self.tokenizer = get_tokenizer("basic_english")   

        root_dir = os.path.join(CHECKPOINT_PATH, "SampleDecoderTask") 
        self.vocab = build_vocab(partial(yield_tokens, self.tokenizer), root_dir + "/vocab_prebuild.pth")
        self.pad = PadTransform(max_length=max_seq_len, pad_value=self.vocab['<pad>'])

        # inputs =torch.stack([decorated_text_to_ids(text, None, self.pad, self.tokenizer, self.vocab) 
        #                      for text in texts])        
        # labels = shift_tokens_right(inputs, self.vocab['<pad>'])
        # self.data = list(zip(inputs, labels))

        inputs, targets = [], []
        for idx, text in enumerate(texts):
            if max_size is not None and idx >= max_size:
                break
            input = decorated_text_to_ids(text, truncate=None, pad=None, tokenizer=self.tokenizer, vocab=self.vocab)
            target = torch.cat([input.clone()[1:], torch.tensor([self.vocab['<eos>']])])
            inputs.append(input)
            targets.append(target)

        self.data = list(zip(inputs, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def prepare_data_loader():
    batch_size = 3
    train_loader = data.DataLoader(SampleDecoderDataset(texts=texts))
    val_loader   = data.DataLoader(SampleDecoderDataset(texts=test_texts))

    return train_loader, val_loader, val_loader


if __name__ == "__main__":

    data_set = SampleDecoderDataset()
    print(data_set[0])

    train_loader, val_loader, _ = prepare_data_loader()
    print(train_loader.dataset[0])
