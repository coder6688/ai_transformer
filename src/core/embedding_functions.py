import torch
import torch.nn as nn
from config.env import device

def embedding_GloVe(vocab, embed_dim=300):
        from torchtext.vocab import GloVe
        glove = GloVe(name='6B', dim=embed_dim) # only 300 available for GloVe
        weights = glove.get_vecs_by_tokens(vocab.get_itos(), lower_case_backup=True).to(device)
        return weights

def embedding_torch(vocab, embed_dim):
    # option: use torch built-in embedding
    # Explicitly initialize embedding weights on MPS first
    # must create the nn.Embedding inside the device context,
    # otherwise it will not send to device, but stay in cpu.
    with torch.device('mps'):
        embed = nn.Embedding(len(vocab), embed_dim)

        if not hasattr(embed.weight, 'mps_storage'):
            # Force storage allocation
            embed.weight.data = torch.nn.Parameter(
                torch.empty_like(embed.weight.data).normal_()
            )

        return embed.weight
        
def embedding_word2vec(vocab, embed_dim=300):
    assert embed_dim == 300, 'world2vec embedding dimension must be 300.'

    from gensim.models import KeyedVectors
    word2vec = KeyedVectors.load_word2vec_format('embedding_data/GoogleNews-vectors-negative300.bin', binary=True)
    
    with torch.device('mps'):
        weights = torch.zeros((len(vocab), embed_dim))
        for i, word in enumerate(vocab.get_itos()):
            try: 
                weights[i] = torch.tensor(word2vec[word])
            except KeyError:
                # use random weight for over-vocab word
                weights[i] = torch.normal(mean=0, std=0.1, size=(embed_dim,))

        return weights
