import torch
import torch.nn as nn
import math

class PositionNN(nn.Module):
    
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()

        assert embed_dim % 2 == 0, "Must be even number"
        
        # Create position indices for tokens [0, 1, ..., max_len-1] as tensor of shape [max_seq_len, 1]
        # eg. [[0.], [1.], ..., [max_len-1.]]
        # providing verticle broadcast
        input_position = torch.arange(max_seq_len, dtype=torch.float).view(-1, 1)

        constant = -math.log(10000.0) / embed_dim
        # providing horizontal broadcast
        exponent = torch.arange(0, embed_dim, 2).float() * constant
        frequency = torch.exp(exponent)

        # calculate the angle values for embeded dimension for each token
        # broadcast to [max_seq_len, fea_dim/2]
        angle_radians = input_position * frequency

        # position encoding
        pos_encoding = torch.zeros(max_seq_len, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(angle_radians)
        pos_encoding[:, 1::2] = torch.cos(angle_radians)

        # add batch dimension
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, x):
        # x has shape [batch_size, seq_len, fea_dim]
        # pos_encoding has shape [1, max_seq_len, fea_dim]
        # slicing to [1, seq_len, fea_dim]
        # This broadcasts over batch_size
        seq_len = x.size(1)

        # pos_encoding[:, :seq_lin] meaning:
        # 1. it specifies how to take elements along two dimensions: first one and second one
        # 2. For first dimension, we want to take all elements, eg. all batchs, which actually just 1
        # 3. For second dimension, we want to take the first `seq_len` elements, which is exactly what we need, eg. 10 words
        # 4. This results in a tensor of shape [1, seq_len, fea_dim]
        # 5. Once added to x, it will broadcast over batch_size
        x = x + self.pos_encoding[:, :seq_len]
        return x
    

if __name__ == "__main__":
    # visualize the position encoding
    import matplotlib.pyplot as plt

    posNN = PositionNN(embed_dim=24, max_seq_len=100)

    pos_encoding = posNN.pos_encoding.squeeze().T.cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
    pos = ax.imshow(pos_encoding, cmap="RdGy", extent=(1,pos_encoding.shape[1]+1,pos_encoding.shape[0]+1,1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1]+[i*10 for i in range(1,1+pos_encoding.shape[1]//10)])
    ax.set_yticks([1]+[i*10 for i in range(1,1+pos_encoding.shape[0]//10)])
    plt.show()