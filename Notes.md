## How to run trainning
```
source <your_virtual_python_env>
python src/training/training_reverse_task.py
python src/training/training_shifted_seq_task.py
python src/training/training_imdb_review_task.py
```
## How to run tensorboard
```
tensorboard --logdir=/Users/rtong/workspace/saved_models/ShakespeareDecoderTask/lightning_logs/version_???
```
## How to run test
```
source <your_virtual_python_env>
python src/testing/test_imdb_review_task.py
```

## Word2Vec download
```
# Direct download (2.2GB compressed, 3.6GB decompressed)
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

# Decompress
gunzip GoogleNews-vectors-negative300.bin.gz

# save it to embedding_data folder for training to load
```

# Download Shakespear's play
```
pip install kaggle
   kaggle datasets download -d kingburrito666/shakespeare-plays
   unzip shakespeare-plays.zip
```

## Local Python Virtual Environment Library List
torch and torchtext must have compatible version.
Local Environment: Mac Mini M4

Python 3.10.14
```
aiohttp            3.11.12
Brotli             1.1.0
cffi               1.17.1
colabcode          0.1.1
colorama           0.4.6
dill               0.3.9
en_core_web_sm     3.8.0
gdown              5.2.0
gensim             4.3.3
gmpy2              2.1.5
h2                 4.2.0
importlib_metadata 8.6.1
ipykernel          6.29.5
pickleshare        0.7.5
pip                25.0.1
portalocker        3.0.0
PySocks            1.7.1
pytorch-lightning  2.5.0.post0
seaborn            0.13.2
spacy              3.8.4
tensorboard        2.19.0
torchdata          0.9.0
torchtext          0.18.0
wheel              0.45.1
zstandard          0.23.0
```