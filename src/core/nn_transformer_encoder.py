import torch.nn as nn

from core.nn_encoder_block import EncoderBlockNN

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **encoder_args):
        super(TransformerEncoder, self).__init__()

        # Multiple layers of encoders
        self.layers = nn.ModuleList([EncoderBlockNN(**encoder_args)] * num_layers)

    def forward(self, x, mask=None):
        # Pass the input through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        return x
    
    def compute_attention_mats(self, x, mask=None):
        attention_mats = []  # List to store attention matrices
        for layer in self.layers:
            # call the attentionNN of each layer, eg. __call__ and call forward()
            _, attn_mat = layer.attentionNN(x, mask, return_attention_mat=True)
            attention_mats.append(attn_mat)

        return attention_mats
