import torch
import torch.nn as nn
import pytorch_lightning as pl
import dill  # Extended pickle for function serialization

from core.nn_position import PositionNN
from core.nn_transformer_encoder import TransformerEncoder

class TransformerTrainer(pl.LightningModule):
    def __init__(self, optimizer_config=None, embedding_func=None, dropout=0.0, input_dropout=0.0, **kwargs):
        super().__init__()
        
        # Store optimizer config separately
        self.optimizer_config = optimizer_config

        # Customized embedding function
        self.embedding_func = embedding_func

        # Customized summary writer
        self.summary_writer = kwargs["summary_writer"] if "summary_writer" in kwargs else None
        
        # Explicitly ignore non-serializable parameter
        self.save_hyperparameters(ignore=['optimizer_config', 'embedding_func', 'summary_writer'])

        self.embed_layer = nn.Embedding(self.hparams.input_dim, self.hparams.embed_dim)
        with torch.no_grad():
            self.embed_layer.weight.copy_(embedding_func())
        # freeze by default 
        self.embed_layer.weight.requires_grad = False if (hasattr(self.hparams, "embed_freeze") and self.hparams.embed_freeze) else True
        
        self.input_layer = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.embed_dim, self.hparams.model_dim)
        )
        
        self.position_layer = PositionNN(model_dim=self.hparams.model_dim)

        encoder_args = {
            "model_dim" : self.hparams.model_dim, 
            "fea_dim" : self.hparams.fea_dim,  
            "mlp_dim" : self.hparams.mlp_dim, 
            "num_heads" : self.hparams.num_heads, 
            "dropout" : self.hparams.dropout
        }
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers, **encoder_args)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.embed_dim)  # make sure it goes back to embedding space
        )

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
            self.criterion = nn.CrossEntropyLoss()

        self.custom_functions = {
            'embedding_func': self.embedding_func,
            'optimizer_config': self.optimizer_config
        }

    def _save_custom_functions(self, path):
        """Save function implementations separately"""
        with open(path, 'wb') as f:
            dill.dump(self.custom_functions, f)

    @classmethod
    def _load_custom_functions(cls, path):
        """Load serialized functions"""
        with open(path, 'rb') as f:
            return dill.load(f)

    def save_checkpoint(self, path):
        """Full model save with function serialization"""
        # Save main model
        torch.save({
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'pytorch-lightning_version': pl.__version__
        }, path)
        
        # Save functions in parallel file
        self._save_custom_functions(path + '.func')

    @classmethod
    def load_from_checkpoint(cls, path):
        """Load model with function restoration"""
        # Load main model
        checkpoint = torch.load(path, map_location='cpu')
        funcs = cls._load_custom_functions(path + '.func')
        model = cls(**checkpoint['hyper_parameters'], **funcs)
        model.load_state_dict(checkpoint['state_dict'])
        
        # # Load and attach functions
        # funcs = cls._load_custom_functions(path + '.func')
        # model.custom_functions.update(funcs)
        
        return model

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

    def configure_optimizers(self):
        if self.optimizer_config is None:
            return super().configure_optimizers()
        else:
            return self.optimizer_config(self.parameters(), **self.hparams)
        
    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError
    
    def test_step(self, *args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def get_attention_mats(self, x, mask=None, need_pos_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_layer(x)
        if need_pos_encoding:
            x = self.position_layer(x)
        attention_mats = self.transformer.compute_attention_mats(x, mask=mask)
        return attention_mats
