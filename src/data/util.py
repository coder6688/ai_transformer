import re
import pathlib

import torch
from torchtext.vocab import build_vocab_from_iterator


def shakespeare_tokenizer(text):
    # Preserve contractions and archaic forms
    text = re.sub(r"'(Tis|Twas|Twere|Twill|Twould)(\W)", r" ' \1\2", text, flags=re.IGNORECASE)
    
    # Tokenization rules
    return re.findall(r"""
        \w+['’]\w+          # Contractions: 'tis, 'twas
        |\w+-\w+            # Hyphenated words: self-born
        |[A-Za-z]+          # Standard words
        |[!?.,;:’'"()]      # Punctuation
        |\n                 # Newline preservation
    """, text, re.X)

def build_vocab(yield_tokens, vocab_file=None):

    path = pathlib.Path(vocab_file)

    # re-generate in case tokenizer etc change 
    if path.exists() and path.is_file():
        return torch.load(vocab_file)

    vocab = build_vocab_from_iterator(
        yield_tokens(),
        specials=['<pad>', '<unk>', '<sos>', '<eos>'],
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])
    torch.save(vocab, vocab_file)
    return vocab

def text_to_tensor(vocab, tokenizer, text, seq_len=512):
    # only use the first seq_len of words for prediction
    return torch.tensor([vocab[token] for idx, token in enumerate(tokenizer(text))
                            if idx < seq_len], dtype=torch.long)

def collate_batch(batch, pad_id=0, on_left=True):
    max_len_input  = max([len(x[0]) for x in batch])
    max_len_label = max([len(x[1]) for x in batch])

    inputs, labels = [], []
    for input, label in batch:
        if on_left:
            # left-padding is always easy to pick out the prediction
            padded_input = torch.nn.functional.pad(input, (max_len_input - input.size(0), 0), value=pad_id)
            inputs.append(padded_input)

            padded_label = torch.nn.functional.pad(label,(max_len_label - label.size(0), 0), value=pad_id)
            labels.append(padded_label)
        else:
            # have to specifically figure out which position to pick out the prediction
            padded_input = torch.nn.functional.pad(input, (0, max_len_input - input.size(0)), value=pad_id)
            inputs.append(padded_input)

            padded_label = torch.nn.functional.pad(label,(0, max_len_label - label.size(0)), value=pad_id)
            labels.append(padded_label)


    return torch.stack(inputs), torch.stack(labels)

def decorated_text_to_ids(text, truncate, pad, tokenizer, vocab, add_soss_eos=False):
    result = vocab(tokenizer(text))
    if add_soss_eos:
        result =[vocab['<sos>']] + result + [vocab['<eos>']]
    if truncate is not None:
        result = truncate(result)
    if pad is not None:
        result = pad(torch.tensor(result))
    return torch.tensor(result)

def shift_tokens_right(input_ids, pad_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, :-1] = input_ids[:, 1:].clone()
    shifted[:, -1] = pad_token_id
    return shifted

def text_to_ids(vocab, tokenizer, x):
    return vocab(tokenizer(x))
