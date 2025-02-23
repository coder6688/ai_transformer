import torch
import torch.nn as nn

from core.compute_attention import compute_attention


class MultiHeadAttentionNN(nn.Module):
    """
    Understand what is the head in multi-head attention:
    In multi-head attention, the embedding dimension is split across multiple heads. 
    Each head operates on a smaller subspace.
    Each head has its own query, key, value, and compute independently.
    Finally, the outputs of all heads are concatenated back to the original embedding/feature dimension.

    Example:
    embed_dim = 512
    num_heads = 8
    head_dim = 512 // 8 = 64
    How it's used:
    1. Query/Key/Value projections are split into num_heads vectors of size head_dim
    2. Allows each head to learn different attention patterns
    3. Final outputs are concatenated back to fea_dim   
    4. Additional linear NN is applied to the final output to mix the different heads' info
 
    dimension flow:
        Input to MHA: [batch_size, seq_len, input_dim], eg. input_dim = one-hot encoding dim of 100,000 vacab
        After QKV projection: [batch_size, seq_len, 3*fea_dim], eg. fea_dim could be much smaller as 512
        After split heads: [batch_size, heads, seq_len, head_dim]
        After attention: [batch_size, heads, seq_len, head_dim]
        After concat: [batch_size, seq_len, fea_dim]  # heads*head_dim = fea_dim
        After mixing_heads_proj: [batch_size, seq_len, fea_dim]

    input shape: [batch_size, seq_len, embed_dim]
    output shape: [batch_size, seq_len, fea_dim]
    """

    def __init__(self, model_dim, fea_dim, num_heads, dropout=0.1):
        # NOTE: here we allow to project the embedding space into a different feature space!
        assert fea_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"
        super(MultiHeadAttentionNN, self).__init__()

        self.num_heads = num_heads
        self.fea_dim = fea_dim
        self.head_dim = fea_dim // num_heads

        # Create a linear layer for projecting the input to the q/k/v concatenated vectors
        # input: [batch_size, seq_len, embed_dim]
        # output: [batch_size, seq_len, 3*fea_dim]   
        self.qkv_linearNN = nn.Linear(model_dim, 3*fea_dim, bias=False)  # test with bias=False

        # Create a linear layer for mixing the outputs of the heads
        # input: [batch_size, seq_len, fea_dim]
        # output: [batch_size, seq_len, model_dim]  
        # IMPORTANT: output must have the same shape as input, as the output will add the input(x) again 
        self.output_linearNN = nn.Linear(fea_dim, model_dim)  
           

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.qkv_linearNN.weight)
        nn.init.xavier_normal_(self.output_linearNN.weight)

    def forward(self, x, mask=None, return_attention_mat=False):
        batch_size, seq_len, model_dim = x.size()

        # Project the input to the q/k/v concatenated vectors
        # input: [batch_size, seq_len, model_dim]
        # output: [batch_size, seq_len, 3*fea_dim]   
        qkv_mat = self.qkv_linearNN(x)

        # reshape for multi-heads to [batch_size, seq_len, num_heads, 3*head_dim]
        # note: fea_dim = num_heads * head_dim
        qkv_mat = qkv_mat.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)

        # re-arrange the matrix to [batch_size, num_heads, seq_len, 3*head_dim]
        qkv_mat = qkv_mat.transpose(1, 2)

        # split into q, k, v
        q, k, v = torch.chunk(qkv_mat, 3, dim=-1)

        # compute attention matrix and attended value
        attention_mat, attended_value = compute_attention(q, k, v, mask=mask)
        # re-arrange attended_v to [batch_size, seq_len, num_heads, head_dim]
        attended_value = attended_value.transpose(1, 2)
        # re-arrange to [batch_size, seq_len, fea_dim]
        attended_value = attended_value.reshape(batch_size, seq_len, self.fea_dim)

        # mixing info from different heads to [batch_size, seq_len, model_dim]
        output = self.output_linearNN(attended_value)

        return output, attention_mat if return_attention_mat else None
