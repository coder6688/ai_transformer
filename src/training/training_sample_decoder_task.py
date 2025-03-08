import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from functools import partial

from config.set_seed import set_seed
from core.embedding_functions import embedding_GloVe, embedding_torch, embedding_word2vec
from core.get_optimizer_lr import get_optimizer_with_lr
from core.decoder_task_trainer import DecoderTaskTrainer
from data.sample_decoder_dataset import prepare_data_loader
from data.util import collate_batch, text_to_tensor
from training.training_common import training


if __name__ == "__main__":
    set_seed()

    train_loader, val_loader, test_loader = prepare_data_loader()

    vocab = train_loader.dataset.vocab
    vocab_size = len(train_loader.dataset.vocab)
    num_layers = 2
    num_heads  = 1
    input_dim  = vocab_size # vocab_size, actually will be done by looking up by ids
    embed_dim  = 300 # default 

    task_model, task_result = training(train_loader, val_loader, test_loader, DecoderTaskTrainer,
                                        task_name="SampleDecoderTask",
                                        padding_mask=True,
                                        embed_freeze=True,
                                        vocab_size = vocab_size,
                                        input_dim=input_dim,
                                        embed_dim=embed_dim,                                        
                                        model_dim=embed_dim,
                                        fea_dim=64,
                                        mlp_dim=256,
                                        num_heads=num_heads,
                                        output_dim=vocab_size,
                                        num_layers=num_layers,
                                        dropout=0,
                                        input_dropout=0,
                                        max_epochs=50,
                                        lr=1e-3,
                                        warmup=50,                                                  
                                        optimizer_config=get_optimizer_with_lr("linear_const_lr"),
                                        embedding_func=partial(embedding_torch, vocab=train_loader.dataset.vocab, embed_dim=embed_dim),
                                        pad_id=vocab['<pad>'],
                                        # logging
                                        log_every_n_steps=5,
                                        #limit_val_batches=0.1,
                                        #val_check_interval=0.5
                                        )
    
    print(f"Val accuracy:  {(100.0 * task_result['val'][0]['test_acc']):4.2f}%")

    if True:
        # Test to see if it can match the test_imdb_review_task.py.
        # It does matach exactly.
        import os
        import torch
        from torchtext.data.utils import get_tokenizer
        from config.env import CHECKPOINT_PATH, device

        model = task_model
        vocab_path = os.path.join(CHECKPOINT_PATH, "SampleDecoderTask", "vocab_prebuild.pth")
        vocab = torch.load(vocab_path)

        tokenizer = get_tokenizer("basic_english")

        texts = ["who are"]

        data = [
            (text_to_tensor(vocab, tokenizer, text), text_to_tensor(vocab, tokenizer, text)) for text in texts
        ]

        inputs, label = collate_batch(data)
        inputs = inputs.to(device)
        model.eval()
        with torch.no_grad():
            inp_data = model.embed_layer(inputs)
            output = model(inp_data)
            probs = torch.sigmoid(output)
            predicted_ids = torch.argmax(probs[:, -1, :], dim=-1)
            text = [vocab.get_itos()[i.item()] for i in predicted_ids]
            
            print(f'prediction: {text}')
