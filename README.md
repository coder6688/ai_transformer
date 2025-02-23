**Important Links**  
> [📌 Project Notes](./Notes.md)


# 1. Overview
```
# Transformer Architecture Experiments

This project implements a bidirectional Transformer encoder for three sequence processing tasks:
1. Numerical Sequence Transformation
   - Sequence reversal (e.g., 1234 → 4321)
   - Sequence shifting (e.g., 1234 → 2341)
   - Visualized self-attention patterns demonstrate the model's contextual learning mechanism
2. IMDB Sentiment Analysis (Binary Classification)
   - Comparative evaluation of embedding strategies:
     - Pretrained Word2Vec
     - GloVe (Global Vectors, 300-dim)
     - Trainable PyTorch embeddings
   - Learning rate scheduler comparison:
     - Original Transformer linear warmup
     - Cosine warmup
     - Constant learning rate
   - Achieved 87% validation accuracy with 4 attention heads

## Key Features
- Multi-head attention implementation with configurable heads (4-8)
- Three learning rate scheduling strategies:
  - Linear warmup (1e-7 → 1e-3 over 4k steps)
  - Cosine annealing with warm restarts
  - Constant learning rate at 5e-4
- Integrated visualization tools:
  - TensorBoard for training metrics tracking
  - Attention matrix heatmaps for model interpretation

## Experimental Insights
1. Attention patterns for numerical tasks show clear diagonal focus positions
2. Trainable embeddings outperformed static embeddings by F1-score
3. Proper learning rate (1e-4 to 1e-3) is crucial
4. 4-head attention achieved best accuracy/throughput balance (78 samples/sec)

## Sample Tests for IMDB Review Sentiment Analysis
1. Reviews
The model achieves comparable training and validation accuracy scores (~85%), 
but qualitative analysis reveals limitations in classifying simple, unseen sentences. 
Interestingly, while quantitative metrics show similar performance across embedding methods, 
PyTorch's native embeddings demonstrate superior relative sentiment scoring compared to 
pre-trained GloVe and Word2Vec embeddings in these tests. Note: here embeddings are frozen
during the training.

Notable observations:
- Randomly initialized PyTorch embeddings outperform pre-trained counterparts in sentiment intensity ranking
- Model struggles with syntactic variations despite high accuracy scores
- Embedding performance gap persists across different random initializations
- Quantitative metrics (accuracy, F1) don't fully capture semantic ranking capabilities

2. Sample setting comparison
    num_layers=4,
    model_dim=16,
    fea_dim=16,
    num_heads=4, # note: fea_dim % num_heads == 0
    mlp_dim=32,
    num_classes=train_loader.dataset.num_categories,
    dropout=0.3,
    input_dropout=0,
```
![Sentiment Test](resources/sample_sentiment_tests.png)

```
3. Embedding Trainable
The trainable parameters are too many given the limit training data. The performances all become similar to each other.
```

![Sentiment Test](resources/sample_sentiment_tests_trainable_embedding.png)

# 2. Component Hierarchy
## Overview
![Overview component Diagram](resources/overview_components.png)

## Data Flow Sequence:
```
Raw Text → Cleaning → Tokenization → Vocabulary Mapping → 
Processed Files → Dataset → DataLoader → Model Input
```

# 3. Sample Applications
```
All tests are run in MAC mini M4
```
## 3.1 Predict shifted number list (shift 1, 3, 5 position): Attention Matrix & Training History
![Attention Matrix](resources/attn_mat_shifted_1.png)
![Attention Matrix](resources/attn_mat_shifted_3.png)
![Attention Matrix](resources/attn_mat_shifted_5.png)

## 3.2 Predict reversed number list: Attention Matrix & Training History
![Attention Matrix](resources/reverse_list.png)


## 3.3 IMDB Review Sentiment Task: Training History
![IMDB Review Task](resources/IMDB_review_sentiment_training.png)
![IMDB Review Task](resources/IMDB_review_sentiment_learning_rate.png)

## An IMDB Sample Training Parameters
![IMDB Sample Task](resources/training_sample_parameters.png)
![IMDB Sample Task](resources/training_sample_batch.png)
![IMDB Sample Task](resources/training_sample_acc_loss.png)
![IMDB Sample Task](resources/training_sample_lr.png)

# 4. IMDB Review Sentiment Task Comparison
## Embedding Effect (GloVe vs Google Word2Vec vs Torch Embedding)
![IMDB Sample Task](resources/training_sample_parameters_4_layers.png)

### GloVe (fixed weights)
![IMDB Sample Task](resources/training_sample_acc_loss_4_layers.png)
### GloVe (unfixed weights)
![IMDB Sample Task](resources/training_sample_acc_loss_4_layers_glove_embed_train.png)

### Word2Vec (fixed weights)
![IMDB Sample Task](resources/training_sample_acc_loss_4_layers_word2vec.png)
### Word2Vec (unfixed weights)
![IMDB Sample Task](resources/training_sample_acc_loss_4_layers_word2vec_embed_train.png)
### Word2Vec Parameters: fixed vs unfixed weights
![IMDB Sample Task](resources/trainable_embedding_world2vec_parameters_false.png)
![IMDB Sample Task](resources/trainable_embedding_world2vec_parameters_true.png)

### Torch.Embedding (fixed weights)
![IMDB Sample Task](resources/training_sample_acc_loss_4_layers_torch_embedding.png)
### Torch.Embedding (unfixed weights)
![IMDB Sample Task](resources/training_sample_acc_loss_4_layers_torch_embedding_train.png)
### Torch.Embedding (trainable parameters)
![IMDB Sample Task](resources/embeding_trainable_parameters.png)

# 5. Tensor Board
```
tensorboard --logdir=logs --port=6006
```
From browser
```
http://localhost:6006
```
