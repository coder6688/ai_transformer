import torch.nn as nn

from core.nn_multihead_attention import MultiHeadAttentionNN

class EncoderBlockNN(nn.Module):
    def __init__(self, model_dim, fea_dim, mlp_dim, num_heads, dropout=0.1):
        super(EncoderBlockNN, self).__init__()

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
        attention, _ = self.attentionNN(x, mask, return_attention_mat=False)

        # Add & Drop & Norm
        # Normalizes activations across the feature dimension (eg. compute within a feature vector)
        # Stabilizes training by reducing internal covariate shift
        # Applied after the residual connection (attention + input)
        # y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
        # γ (scale) and β (bias) are learnable parameters        
        # x = x + self.dropout(output)
        # x = self.norm_after_attentionNN(x)

        # # pre-layernorm dropout (chatGPT)
        # x = x + self.dropout(self.norm_after_attentionNN(attention))
        # output = self.mlpNN(x)
        # x = x + self.dropout(self.norm_after_mlpNN(output))

        # post-layernorm dropout (original paper)
        x = self.norm_after_attentionNN(x + self.dropout(attention))
        output = self.mlpNN(x)
        x = self.norm_after_mlpNN(x + self.dropout(output))

        return x