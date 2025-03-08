import torch
import torch.nn as nn
import pytorch_lightning as pl
import dill  # Extended pickle for function serialization

from core.nn_position import PositionNN

class TransformerTrainer(pl.LightningModule):
    def __init__(self, optimizer_config=None, embedding_func=None, **kwargs):
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

        # after change to embed space
        self.position_layer = PositionNN(embed_dim=self.hparams.embed_dim) 

        # original implementation doesn't have this projection layer
        # experiment with only dropout without Linear. make sure model_dim == embed_dim
        self.input_layer = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            #nn.Linear(self.hparams.embed_dim, self.hparams.model_dim)
        )
        
        #self.position_layer = PositionNN(model_dim=self.hparams.model_dim)

        # encoder/decoder layer to be implemented in subclass

        self.output_layer = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.embed_dim)  # make sure it goes back to embedding space
        )

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
        
        return model

    def configure_optimizers(self):
        if self.optimizer_config is None:
            return super().configure_optimizers()
        else:
            return self.optimizer_config(self.parameters(), **self.hparams)

    def forward(self, x, mask=None, need_pos_encoding=True):
        raise NotImplementedError

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
