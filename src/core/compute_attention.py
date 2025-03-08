import math
import torch


def compute_attention(q, k, v, mask=None):
    """
    Compute the attention matrix between query and key. 
    The attention matrix is a square matrix where each element represents the attention score between a pair of rows in the query and key.
    The attention score is calculated using the dot product of the corresponding rows in the query and key.
    Args:
        q (torch.Tensor): A 2D tensor representing the query. Shape: (batch_size, num_heads, seq_len, hidden_dim)
        k (torch.Tensor): A 2D tensor representing the key. Shape: (batch_size, num_heads, seq_len, hidden_dim)
        v (torch.Tensor): A 2D tensor representing the value. Shape: (batch_size, num_heads, seq_len, hidden_dim)
        mask: boolean flag to mask the attention scores
    Returns:
        torch.Tensor: A 3D tensor representing the attention matrix. Shape: (batch_size, num_heads, seq_len, seq_len)
        torch.Tensor: A 3D tensor of the attended values. Shape: (batch_size, num_heads, seq_len, hidden_dim)
    """
    # Compute the dot product of query and key along the feature dimension
    model_dim = q.size(-1)    

    # torch.matmul(): matrix multiplication supports very general high-dimensional tensors.
    # For 2D tensors, it is equivalent to simple matrix multiplication.
    # For 3D tensors, it is equivalent to a batch of matrix multiplications.
    # For 4D tensors, it is equivalent to a batch of matrix multiplications over number of heads of matrix multiplications.
    # For higher-dimensional tensors, it is equivalent to broadcasting of matrix multiplications over the last two dimensions.
    # eg.  # (..., n,m) @ (..., m,p) → (..., n,p) with broadcasting.
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model_dim)

    if mask is not None:
        attention_scores.masked_fill_(mask, float('-1e18'))

    # Normalize the attention scores using softmax along the seq_len dimension
    # which is column-wise. eg. make the sum of all columns in a row to be 1
    attention_mat = torch.softmax(attention_scores, dim=-1)

    # For all zero rows, the attention_mat values actually become quite large uniform values.
    # These need to be set to tiny number again using mask
    if mask is not None:
        attention_mat = attention_mat.masked_fill(mask, 0.0)

    # compute the transformed value vector to its contextual representation
    # or compute the weighted sum of v
    # attended_v: (batch_size, num_heads, seq_len, d_tensor)
    # v: (batch_size, num_heads, seq_len, d_tensor)
    # Example: assuming for 1 batch only,
    # Each row in the output can be thought as a weighted sum of all rows in v.
    # The weight for each row in v is the attention weight of the row in v to the query.
    # Then the first row in output can be thought as a weighted sum of the first row in v, the second row in v, ... the last row in v.
    # Since the first word cannot get hint from any word after that, so the attention weight from 2nd column to the last column must be 0.
    # This is masking and the reason why the attention weight matrix is a lower triangular matrix.
    # in graphical representation as: V1, V2, V3 are the row vectors in v, 
    # the output is corresponding to the contextual representation of the 1st, 2nd, 3rd word.
    # [1, 0, 0] [v1]
    # [1, 1, 0] [v2]
    # [1, 1, 1] [v3]
    # The first row in output = [1, 0, 0] * [v1, v2, v3] = [v1, 0, 0]
    # The second row in output = [1, 1, 0] * [v1, v2, v3] = [v1+v2, v2, 0]
    # The third row in output = [1, 1, 1] * [v1, v2, v3] = [v1+v2+v3, v2+v3, v3]    
    attended_v = torch.matmul(attention_mat, v)

    return attention_mat, attended_v


if __name__ == "__main__":

    import pytorch_lightning as pl

    seq_len, fea_dim = 3, 2
    pl.seed_everything(0)
    
    q = torch.randn(seq_len, fea_dim)
    k = torch.randn(seq_len, fea_dim)
    v = torch.randn(seq_len, fea_dim)
    print("q", q)
    print("k", k)
    print("v", v)

    # q,k order actually is not important. But due to the implementation of softmax, 
    # q,k order actually makes attention weights different.
    values, attentions = compute_attention(q, k, v)

    print("values: ", values)
    print("attentions: ", attentions)
