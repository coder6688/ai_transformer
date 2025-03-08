import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch.nn as nn

from core.nn_decoder_block import DecoderBlockNN

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **decoder_args):
        super(TransformerDecoder, self).__init__()

        # Multiple layers of encoders
        self.layers = nn.ModuleList([DecoderBlockNN(**decoder_args)] * num_layers)

    def forward(self, x, mask=None):
        # Pass the input through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        return x
    
    def compute_attention_mats(self, x, mask=None):
        attention_mats = []  # List to store attention matrices

        # todo: move mask out
        # Create causal mask
        import torch
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * 0, 
            diagonal=1
        ).to(x.device)

        for layer in self.layers:
            # call the attentionNN of each layer, eg. __call__ and call forward()
            _, attn_mat = layer.attentionNN(x, causal_mask, return_attention_mat=True)
            attention_mats.append(attn_mat)

        return attention_mats


    if __name__ == "__main__":
        import torch
        from config.env import device
        from core.nn_transformer_decoder import TransformerDecoder
        from core.embedding_functions import embedding_torch


        x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]], dtype=torch.float32).unsqueeze_(0).to(device)
        trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]], dtype=torch.float32).unsqueeze_(0).to(device)

        # has to create embedding layer separately here
        input_dim = 9
        embed_dim = 9
        embed_layer = nn.Embedding(input_dim, embed_dim).to(device)

        num_layers = 4
        decoder_args = {
            "model_dim": embed_dim,
            "fea_dim": 12,
            "mlp_dim": 48, #4 * model_dim,
            "num_heads": 4,
            "dropout": 0.3
        }
        model = TransformerDecoder(num_layers, **decoder_args).to(device)

        # output dim = model_dim
        # so that output dim can be stacked as blocks
        output = model(x)
        print(output)