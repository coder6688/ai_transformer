import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # Check for Apple Silicon GPU
        return "mps"
    else:
        return "cpu"

device = get_device()

# Mac M4 has better float32 performance
torch.set_default_dtype(torch.float32)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/"

