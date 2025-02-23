import torch


def text_to_tensor(vocab, tokenizer, text, seq_len=512):
    # only use the first seq_len of words for prediction
    return torch.tensor([vocab[token] for idx, token in enumerate(tokenizer(text))
                            if idx < seq_len], dtype=torch.long)

def collate_batch(batch):
    max_len = max([len(x[0]) for x in batch])

    result_text_ids, result_label = [], []
    for text_ids, label in batch:
        padded_text_ids = torch.nn.functional.pad(text_ids,
                                                  (0, max_len - text_ids.size(0)),
                                                  value=0)
        result_text_ids.append(padded_text_ids)
        result_label.append(label)
    return torch.stack(result_text_ids), torch.stack(result_label)