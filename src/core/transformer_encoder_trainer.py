import torch.nn as nn

from core.transformer_trainer import TransformerTrainer
from core.nn_transformer_encoder import TransformerEncoder

class TransformerEncoderTrainer(TransformerTrainer):
    def __init__(self, optimizer_config=None, embedding_func=None, dropout=0.0, input_dropout=0.0, **kwargs):
        super().__init__(optimizer_config=optimizer_config, embedding_func=embedding_func, dropout=dropout, input_dropout=input_dropout, **kwargs)
        
        encoder_args = {
            "model_dim" : self.hparams.model_dim, 
            "fea_dim" : self.hparams.fea_dim,  
            "mlp_dim" : self.hparams.mlp_dim, 
            "num_heads" : self.hparams.num_heads, 
            "dropout" : self.hparams.dropout
        }
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers, **encoder_args)

        # always create layer here in order for parameters to be registered and train
        # layer dynamically created in forward pass will not be registered and will not be trained

        if not hasattr(self.hparams, "is_binary_classification"):
            self.hparams.is_binary_classification = False

        if self.hparams.is_binary_classification:
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(self.hparams.embed_dim, 1),
                nn.Sigmoid()
            )
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean') #pad index

        self.custom_functions = {
            'embedding_func': self.embedding_func,
            'optimizer_config': self.optimizer_config
        }

    def forward(self, x, mask=None, need_pos_encoding=True):
        x = self.input_layer(x)

        if need_pos_encoding:
            x = self.position_layer(x)

        x = self.transformer(x)
        x = self.output_layer(x)

        if self.hparams.is_binary_classification:
            # Approach 1: pool on average
            x = x.transpose(1, 2) # [batch, seq_len, fea_dim] => [batch, fea_dim, seq_len]: 
                                  # last dimension contains all values for a sentence
                                  # one mostly works with values at the last dimension, 
                                  # so use reshape to make the last dimension values relevant for operation.
                                  # eg. here aggregate the values for one sentence.
            x = self.pool(x).squeeze(-1) # => [batch, fea_dim]
            x = self.classifier(x) # => [batch, 1]

        return x
