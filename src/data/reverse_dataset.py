import torch
import torch.utils.data as data
from functools import partial

class ReverseDataset(data.Dataset):

    def __init__(self, num_categories, seq_len, size):
        """
        num_categories: number of categories in the dataset, 
        eg. dimension of the one-hot vector
        seq_len: length of the sequence
        size: number of samples in the dataset
        """
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        # shape: (size, seq_len)
        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels
    
def prepare_data_loader():
    dataset = partial(ReverseDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(dataset(1000), batch_size=128)
    test_loader  = data.DataLoader(dataset(10000), batch_size=128)

    return train_loader,val_loader, test_loader


if __name__ == "__main__":
    from functools import partial
    # partial is a functional that will fix the first two arguments of ReverseDataset)
    dataset = partial(ReverseDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(dataset(1000), batch_size=128)
    test_loader  = data.DataLoader(dataset(10000), batch_size=128)

    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)
