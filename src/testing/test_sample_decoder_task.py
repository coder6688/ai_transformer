import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import torch.utils.data as data
from torchtext.data.utils import get_tokenizer

from config.env import CHECKPOINT_PATH, device
from core.decoder_task_trainer import DecoderTaskTrainer
from data.sample_decoder_dataset import prepare_data_loader
from data.util import collate_batch, text_to_tensor

# These are needed to restore the model
from core.embedding_functions import embedding_GloVe, embedding_torch, embedding_word2vec

# These are needed to restore the model
train_loader, val_loader, test_loader = prepare_data_loader()

tokenizer = get_tokenizer("basic_english")
vocab_path = os.path.join(CHECKPOINT_PATH, "SampleDecoderTask", "vocab_prebuild.pth")
vocab = torch.load(vocab_path)

model_name = "SampleDecoderTask"
model_path = os.path.join(CHECKPOINT_PATH, f"{model_name}.pth")
model = DecoderTaskTrainer.load_from_checkpoint(model_path)
model.eval()

text = "Love" # "love" # "Transformer" #"Transformers" #"Hello" # "I"

data = [(text_to_tensor(vocab, tokenizer, text), text_to_tensor(vocab, tokenizer, text))]

inputs, label = collate_batch(data)

model.eval()
result = [text]

max_length = 20
with torch.no_grad():
    while True:
        inp_data = model.embed_layer(inputs)
        output = model(inp_data)
        probs = torch.sigmoid(output)
        predicted_id = torch.argmax(probs[:, -1, :], dim=-1).squeeze()
        text = vocab.get_itos()[predicted_id.item()]
        
        if predicted_id == vocab['<unk>'] or predicted_id == vocab['<eos>']: # or predicted_id == vocab['<pad>']:
            print(f'encounter: {text}')
            break

        result.append(text)

        if len(result) > max_length: 
            break

        inputs = torch.cat([inputs, torch.tensor([[predicted_id]])], dim=-1)

print(' '.join(result))
