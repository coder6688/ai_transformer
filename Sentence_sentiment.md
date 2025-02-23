## Sample Tests for IMDB Review Sentiment Analysis
1. Reviews
From below sample tests, it can be seen even though the model's training and validation
accuracy are both around 85%. It still cannot accurately classify some of the simple, but
un-seen sentences.

2.
texts = ["I really enjoyed this movie!",
         "I do not enjoyed this movie!",
         "what a great movie!",
         "what a good movie!",
         "what a bad movie!",
         "what a terrible movie",
         "A worst movie!"]

2.1 Torch Embedding Freeze, 10 iteration, 4_layer:

                                        # Very close to the best one
                                        num_layers=4, # common: 4
                                        model_dim=16,
                                        fea_dim=16,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=32, # note: 4 * fea_dim
    Sample 0 score: 0.6295 
    Sample 1 score: 0.5041 
    Sample 2 score: 0.7248 
    Sample 3 score: 0.5643 
    Sample 4 score: 0.5020 
    Sample 5 score: 0.5021 
    Sample 6 score: 0.5020 

                                        # THE BEST and RATHER ACCURATE (if ignore >=0.5 ) !!!!!
                                        num_layers=4, # common: 4
                                        model_dim=16,
                                        fea_dim=16,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=32, # note: 4 * fea_dim
                                        num_classes=train_loader.dataset.num_categories,
                                        dropout=0.3,
                                        input_dropout=0,
    Sample 0 score: 0.7176 
    Sample 1 score: 0.5070 
    Sample 2 score: 0.7271 
    Sample 3 score: 0.6709 
    Sample 4 score: 0.5078 
    Sample 5 score: 0.5078 
    Sample 6 score: 0.5077 


2.2 Word2Vec Embedding Freeze, 10 iteration
    Sample 0 score: 0.7302 
    Sample 1 score: 0.7301 
    Sample 2 score: 0.7302 
    Sample 3 score: 0.7301 
    Sample 4 score: 0.5010 
    Sample 5 score: 0.7285 
    Sample 6 score: 0.5030 

                                        num_layers=4, # common: 4
                                        model_dim=16,
                                        fea_dim=16,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=32, # note: 4 * fea_dim
                                        num_classes=train_loader.dataset.num_categories,
                                        dropout=0.3,
                                        input_dropout=0,
    Val accuracy:  83.74%
    Sample 0 score: 0.7280 
    Sample 1 score: 0.7280 
    Sample 2 score: 0.7280 
    Sample 3 score: 0.7279 
    Sample 4 score: 0.5803 
    Sample 5 score: 0.7256 
    Sample 6 score: 0.7091                                             

2.3 GloVe Freeze, 10 iteration
                                        num_layers=4, # common: 4
                                        model_dim=32,
                                        fea_dim=32,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=128, # note: 4 * fea_dim
    Sample 0 score: 0.7308 
    Sample 1 score: 0.7299 
    Sample 2 score: 0.7308 
    Sample 3 score: 0.7305 
    Sample 4 score: 0.5012 
    Sample 5 score: 0.5615 
    Sample 6 score: 0.5172 

                                        # this can be exactly replicated
                                        num_layers=4, # common: 4
                                        model_dim=16,
                                        fea_dim=16,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=64, 

    # droptout effect
    # dropout = 0.0
    Sample 0 score: 0.7304 
    Sample 1 score: 0.7275 
    Sample 2 score: 0.7303 
    Sample 3 score: 0.7231 
    Sample 4 score: 0.5007 
    Sample 5 score: 0.5013 
    Sample 6 score: 0.5009 

    # dropout = 0.05
    Sample 0 score: 0.7048 
    Sample 1 score: 0.5302 
    Sample 2 score: 0.5335 
    Sample 3 score: 0.5004 
    Sample 4 score: 0.5002 
    Sample 5 score: 0.5002 
    Sample 6 score: 0.5002 

    # dropout = 0.1
    Sample 0 score: 0.7033 
    Sample 1 score: 0.5255 
    Sample 2 score: 0.6640 
    Sample 3 score: 0.5011 
    Sample 4 score: 0.5008 
    Sample 5 score: 0.5007 
    Sample 6 score: 0.5007 

    # dropout = 0.2
    Sample 0 score: 0.7012 
    Sample 1 score: 0.5120 
    Sample 2 score: 0.5364 
    Sample 3 score: 0.5013 
    Sample 4 score: 0.5009 
    Sample 5 score: 0.5010 
    Sample 6 score: 0.5010 

    # dropout = 0.3 (best so far)
    Sample 0 score: 0.7265 
    Sample 1 score: 0.5098 
    Sample 2 score: 0.6904 
    Sample 3 score: 0.5022 
    Sample 4 score: 0.5017 
    Sample 5 score: 0.5018 
    Sample 6 score: 0.5018 

    # dropout = 0.4 (worse)
    Sample 0 score: 0.5359 
    Sample 1 score: 0.5059 
    Sample 2 score: 0.5096 
    Sample 3 score: 0.5054 
    Sample 4 score: 0.5051 
    Sample 5 score: 0.5050 
    Sample 6 score: 0.5051 

    # dropout = 0.3, input_dropout = 0.1 (close to the above best, second best)
    Sample 0 score: 0.6988 
    Sample 1 score: 0.5053 
    Sample 2 score: 0.5835 
    Sample 3 score: 0.5033 
    Sample 4 score: 0.5023 
    Sample 5 score: 0.5024 
    Sample 6 score: 0.5024 

 # dropout = 0.3, input_dropout = 0.3
    Sample 0 score: 0.5833 
    Sample 1 score: 0.5107 
    Sample 2 score: 0.5160 
    Sample 3 score: 0.5050 
    Sample 4 score: 0.5047 
    Sample 5 score: 0.5044 
    Sample 6 score: 0.5043 


                                        num_layers=2, # common: 4
                                        model_dim=8,
                                        fea_dim=8,
                                        num_heads=2, # note: fea_dim % num_heads == 0
                                        mlp_dim=32, 
    Sample 0 score: 0.7221 
    Sample 1 score: 0.7142 
    Sample 2 score: 0.7192 
    Sample 3 score: 0.7201 
    Sample 4 score: 0.5001 
    Sample 5 score: 0.5001 
    Sample 6 score: 0.5001 

                                        # not learning to classify well
                                        num_layers=4, # common: 4
                                        model_dim=8,
                                        fea_dim=8,
                                        num_heads=2, # note: fea_dim % num_heads == 0
                                        mlp_dim=32,                                    
    Sample 0 score: 0.7221 
    Sample 1 score: 0.5038 
    Sample 2 score: 0.5676 
    Sample 3 score: 0.5025 
    Sample 4 score: 0.5004 
    Sample 5 score: 0.5004 
    Sample 6 score: 0.5004 
                                        # worst result
                                        num_layers=4, # common: 4
                                        model_dim=16,
                                        fea_dim=16,
                                        num_heads=4, # note: fea_dim % num_heads == 0
                                        mlp_dim=32, 
    Sample 0 score: 0.7308 
    Sample 1 score: 0.7308 
    Sample 2 score: 0.7308 
    Sample 3 score: 0.7308 
    Sample 4 score: 0.7307 
    Sample 5 score: 0.7308 
    Sample 6 score: 0.7308 

                                        # worst result, val at 0.75%
                                        num_layers=4, # common: 4
                                        model_dim=4,
                                        fea_dim=4,
                                        num_heads=1, # note: fea_dim % num_heads == 0
                                        mlp_dim=32,
    Sample 0 score: 0.7303 
    Sample 1 score: 0.7303 
    Sample 2 score: 0.7303 
    Sample 3 score: 0.7303 
    Sample 4 score: 0.7209 
    Sample 5 score: 0.7294 
    Sample 6 score: 0.7228 
```
