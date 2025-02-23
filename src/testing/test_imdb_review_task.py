import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import torch.utils.data as data
from torchtext.data.utils import get_tokenizer

from config.env import CHECKPOINT_PATH
from data.imdb_dataset import prepare_data_loader
from core.task_trainer import TaskTrainer
from data.util import collate_batch, text_to_tensor

# These are needed to restore the model
from core.embedding_functions import embedding_GloVe, embedding_torch, embedding_word2vec

# These are needed to restore the model
train_loader, val_loader, test_loader = prepare_data_loader()

tokenizer = get_tokenizer("basic_english")
vocab_path = os.path.join(CHECKPOINT_PATH, "IMDBReviewTask", "vocab_prebuild.pth")
vocab = torch.load(vocab_path)

model_name = "IMDBReviewTask"
model_path = os.path.join(CHECKPOINT_PATH, f"{model_name}.pth")
model = TaskTrainer.load_from_checkpoint(model_path)
model.eval()

texts = ["I really enjoyed this movie!",
         "I do not enjoyed this movie!",

         "what a great movie!",
         "what a good movie!",

         "what a bad movie!",
         "what a terrible movie",
         "A worst movie!",
         "I hate this movie."
         ]

data = [(text_to_tensor(vocab, tokenizer, text), torch.tensor(0, dtype=torch.float32)) for text in texts]

inputs, label = collate_batch(data)
model.eval()
with torch.no_grad():
    inp_data = model.embed_layer(inputs)
    output = model(inp_data)
    probs = torch.sigmoid(output.squeeze())

    for i, prob in enumerate(probs):
        print(f"Sample {i} score: {prob:.4f}")
