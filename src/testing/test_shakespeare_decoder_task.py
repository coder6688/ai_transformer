import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import torch.utils.data as data

from config.env import CHECKPOINT_PATH, device
from core.decoder_task_trainer import DecoderTaskTrainer
from data.shakespeare_dataset import prepare_data_loader
from data.util import collate_batch, shakespeare_tokenizer, text_to_tensor

# These are needed to restore the model
from core.embedding_functions import embedding_GloVe, embedding_torch, embedding_word2vec

# These are needed to restore the model
train_loader, val_loader, test_loader = prepare_data_loader()

tokenizer = shakespeare_tokenizer
vocab_path = os.path.join(CHECKPOINT_PATH, "ShakespeareDecoderTask", "vocab_prebuild.pth")
vocab = torch.load(vocab_path)

model_name = "ShakespeareDecoderTask"
model_path = os.path.join(CHECKPOINT_PATH, f"{model_name}.pth")
model = DecoderTaskTrainer.load_from_checkpoint(model_path)
model.eval()

texts = ["The edge of war", "To be", "So shaken", "let's marvel"]

for text in texts:
    inputs = text_to_tensor(vocab, tokenizer, text, seq_len=80)
    inputs = inputs.tolist()
    #inputs.insert(0, vocab['<sos>'])
    inputs = torch.tensor(inputs, dtype=torch.long)

    input_len = len(inputs)
    data = [(inputs, inputs)]
    inputs, label = collate_batch(data)

    model.eval()
    result = [text]

    max_length = 200
    top_n = 3
    top_pct = 0.95
    sample_approach = 3
    with torch.no_grad():
        while True:
            inp_data = model.embed_layer(inputs)
            output = model(inp_data)
            probs = torch.sigmoid(output)

            match sample_approach:
                case 1:
                    predicted_id = torch.argmax(probs[:, -1, :], dim=-1).squeeze()

                case 2:
                    _, top_n_idx = torch.topk(probs[:, -1, :], top_n, dim=-1)
                    random_idx = torch.randint(high=top_n-1, size=(1,))
                    predicted_id = top_n_idx.squeeze_().tolist()[random_idx]

                case 3:
                    max_value, _ = torch.max(probs[:, -1, :], dim=-1)                    
                    top_n_idx = torch.where((probs[:, -1, :] >= top_pct * max_value.unsqueeze(-1)))[1]
                    random_idx = torch.randint(high=len(top_n_idx), size=(1,))
                    predicted_id = top_n_idx.squeeze_().tolist()[random_idx]

            text = vocab.get_itos()[predicted_id]
            
            if predicted_id not in [vocab['<pad>'], vocab['<eos>']]:
                result.append(text)

            if len(result) > max_length: 
                break

            inputs = torch.cat([inputs, torch.tensor([[predicted_id]])], dim=-1)

    print(f'### {result[0]}...')
    print(' '.join(result))
    print('\n')
