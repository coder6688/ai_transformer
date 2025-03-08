import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from core.nn_multihead_attention import MultiHeadAttentionNN

class DecoderBlockNN(nn.Module):
    def __init__(self, model_dim, fea_dim, mlp_dim, num_heads, dropout=0.1):
        super(DecoderBlockNN, self).__init__()

        # Attention layer
        self.attentionNN = MultiHeadAttentionNN(model_dim=model_dim, fea_dim=fea_dim, num_heads=num_heads, dropout=dropout)

        # Two-layer Multi-layer perceptron (MLP)
        self.mlpNN = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            #nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, model_dim)  # make sure the output has the same as size of input
        )

        # Dropout is a regularization technique that randomly drops out some neurons during training to prevent overfitting.
        # Each element has probability dropout of being set to 0
        # Surviving elements are scaled by 1/(1-dropout) to maintain mean
        self.dropout = nn.Dropout(dropout)

        # normalization after attention and mlp layers
        self.norm_after_attentionNN = nn.LayerNorm(model_dim)
        self.norm_after_mlpNN = nn.LayerNorm(model_dim)

    def forward(self, x, mask=None):
        # Attention layer
        # TODO: hardcode here for mask the upper triangle for now
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        ) 

        if mask is not None:
            causal_mask = causal_mask | mask
        attention, _ = self.attentionNN(x, causal_mask, return_attention_mat=False)

        # Add & Drop & Norm
        # Normalizes activations across the feature dimension (eg. compute within a feature vector)
        # Stabilizes training by reducing internal covariate shift
        # Applied after the residual connection (attention + input)
        # y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
        # γ (scale) and β (bias) are learnable parameters        
        # x = x + self.dropout(output)
        # x = self.norm_after_attentionNN(x)

        # pre-layernorm dropout (chatGPT)
        x = x + self.dropout(self.norm_after_attentionNN(attention))
        output = self.mlpNN(x)
        x = x + self.dropout(self.norm_after_mlpNN(output))

        # # post-layernorm dropout (original paper)
        # x = self.norm_after_attentionNN(x + self.dropout(attention))
        # output = self.mlpNN(x)
        # x = self.norm_after_mlpNN(x + self.dropout(output))

        return x
    

    if __name__ == "__main__":
        import torch
        from config.env import device
        from core.nn_decoder_block import DecoderBlockNN
        from core.embedding_functions import embedding_torch


        x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]], dtype=torch.float32).unsqueeze_(0).to(device)
        trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]], dtype=torch.float32).unsqueeze_(0).to(device)

        # has to create embedding layer separately here
        input_dim = 9
        embed_dim = 9
        embed_layer = nn.Embedding(input_dim, embed_dim).to(device)

        model_dim = embed_dim
        fea_dim = 12
        mlp_dim = 4 * model_dim
        num_heads = 4
        dropout = 0.3
        model = DecoderBlockNN(model_dim, fea_dim, mlp_dim, num_heads, dropout).to(device)

        # output dim = model_dim
        # so that output dim can be stacked as blocks
        output = model(x)
        print(output)