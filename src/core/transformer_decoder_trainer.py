import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch.nn as nn

from core.transformer_trainer import TransformerTrainer
from core.nn_transformer_decoder import TransformerDecoder

class TransformerDecoderTrainer(TransformerTrainer):
    def __init__(self, optimizer_config=None, embedding_func=None, **kwargs):
        super().__init__(optimizer_config=optimizer_config, embedding_func=embedding_func, **kwargs)
        
        decoder_args = {
            "model_dim" : self.hparams.model_dim, 
            "fea_dim" : self.hparams.fea_dim,  
            "mlp_dim" : self.hparams.mlp_dim, 
            "num_heads" : self.hparams.num_heads, 
            "dropout" : self.hparams.dropout
        }
        self.transformer = TransformerDecoder(num_layers=self.hparams.num_layers, **decoder_args)

        self.logit_output_layer = nn.Linear(self.hparams.embed_dim, kwargs["vocab_size"])
        self.criterion = nn.CrossEntropyLoss()

        self.custom_functions = {
            'embedding_func': self.embedding_func,
            'optimizer_config': self.optimizer_config
        }

    def forward(self, x, mask=None, need_pos_encoding=True):
        ''' for now mask is source padding mask, which apply to all q/k/v the same way assuming 
            same dimensionality of q/k/v. eg. shape ([B, S, S]). These masks must be created and
            passed in as parameters.
            casual_mask will be hardcoded implemented.
        '''

        # position encoding on embedding space: embed_dim
        if need_pos_encoding:
            x = self.position_layer(x)

        # embed_dim => model_dim
        x = self.input_layer(x)

        x = self.transformer(x, mask)

        # model_dim => embed_dim
        x = self.output_layer(x)

        # project to vocab space
        x = self.logit_output_layer(x)

        return x


    if __name__ == "__main__":
        import torch
        from config.env import device
        from core.transformer_decoder_trainer import TransformerDecoderTrainer
        from core.embedding_functions import embedding_torch
        from core.get_optimizer_lr import get_optimizer_with_lr
        from functools import partial


        # token ids
        # shape: [2, 9] eg. [batch, seq_len] 
        x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]], dtype=torch.long).to(device)
        # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]], dtype=torch.long).to(device)
        trg = torch.tensor([[5, 6, 4, 3, 9, 5, 2, 0, 0], [8, 7, 3, 4, 5, 6, 7, 2, 0]], dtype=torch.long).to(device)

        vocab_size = 10
        decoder_args = {
            "seq_len": 9,
            "input_dim": 9,
            "embed_dim": 12,
            "model_dim": 10,
            "fea_dim": 12,
            "mlp_dim": 48, #4 * model_dim,
            "output_dim": vocab_size,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.3,
            "input_dropout": 0.0
        }

        optimizer_config=get_optimizer_with_lr("linear_const_lr")
        embedding_func=partial(embedding_torch, vocab=range(9), embed_dim=decoder_args["embed_dim"])

        trainer = TransformerDecoderTrainer(optimizer_config=optimizer_config, 
                                            embedding_func=embedding_func,
                                            vocab_size=10, # 0-9
                                            **decoder_args).to(device)

        # embed the batch sequence
        embed_layer = nn.Embedding(decoder_args["input_dim"], decoder_args["embed_dim"]).to(device)
        x = embed_layer(x)

        # run transformer
        logit_output = trainer(x, mask=None)

        # compute the predicted token
        next_token_logits = logit_output[:, -1, :]
        predicted_indices = torch.argmax(next_token_logits, dim=-1)

        print(f"prediction(each number for a batch): {predicted_indices}")

