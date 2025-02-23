import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
from functools import partial
import torch.utils.data as data

from config.set_seed import set_seed
from core.get_optimizer_lr import get_optimizer_with_lr
from core.embedding_functions import embedding_GloVe, embedding_torch, embedding_word2vec
from data.imdb_dataset import prepare_data_loader
from data.util import collate_batch, text_to_tensor
from training.training_common import training

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def embedding_func(EMBEDDING_CHOICE=0, embed_dim=300):

    match EMBEDDING_CHOICE: 
        case 0:
            return embedding_GloVe(train_loader.dataset.vocab, embed_dim)
        case 1:
            return embedding_torch(train_loader.dataset.vocab, embed_dim)
        case 2:
            return embedding_word2vec(train_loader.dataset.vocab, embed_dim)
    

if __name__ == "__main__":
    # make sure runs can be replicated.
    set_seed()
    
    train_loader, val_loader, test_loader = prepare_data_loader()

    max_epochs = 10
    print("max_iters: ", max_epochs*len(train_loader))

    input_dim = len(train_loader.dataset.vocab.get_itos())
    embed_dim = 300
    task_model, task_result = training(train_loader, val_loader, test_loader,
                                        task_name="IMDBReviewTask",
                                        seq_len=512, # max-len of a sequence/sentence
                                        is_binary_classification=True,
                                        embed_freeze=True,
                                        input_dim=input_dim,
                                        embed_dim=embed_dim,
                                        num_layers=4, # common: 4
                                        model_dim=16,
                                        fea_dim=16,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=32, # note: 4 * fea_dim
                                        num_classes=train_loader.dataset.num_categories,
                                        dropout=0.3,
                                        input_dropout=0,
                                        max_epochs=max_epochs,
                                        lr=5e-4,
                                        warmup=50,
                                        optimizer_config=get_optimizer_with_lr("linear_const_lr"),
                                        embedding_func=partial(embedding_func, EMBEDDING_CHOICE=2, embed_dim=embed_dim),
                                        )
    
    print(f"Val accuracy:  {(100.0 * task_result['val'][0]['test_acc']):4.2f}%")
    # Disable due to same validation and testing data set
    #print(f"Test accuracy: {(100.0 * task_result['test'][0]['test_acc']):4.2f}%")
    
    if True:
        # Test to see if it can match the test_imdb_review_task.py.
        # It does matach exactly.
        import torch
        from torchtext.data.utils import get_tokenizer
        from config.env import CHECKPOINT_PATH, device

        model = task_model
        vocab_path = os.path.join(CHECKPOINT_PATH, "IMDBReviewTask", "vocab_prebuild.pth")
        vocab = torch.load(vocab_path)

        tokenizer = get_tokenizer("basic_english")

        texts = ["I really enjoyed this movie!",
                 "I do not enjoyed this movie!",
                 "what a great movie!",
                 "what a good movie!",
                 "what a bad movie!",
                 "what a terrible movie",
                 "A worst movie!"]

        data = [
            (text_to_tensor(text), torch.tensor(0, dtype=torch.float32)) for text in texts
        ]

        inputs, label = collate_batch(data)
        inputs = inputs.to(device)
        model.eval()
        with torch.no_grad():
            inp_data = model.embed_layer(inputs)
            output = model(inp_data)
            probs = torch.sigmoid(output.squeeze())

            for i, prob in enumerate(probs):
                print(f"Sample {i} score: {prob:.4f}")
