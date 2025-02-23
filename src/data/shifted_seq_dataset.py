import torch
import torch.utils.data as data

class ShiftedSeqDataset(data.Dataset):

    def __init__(self, num_categories, seq_len, size, shift=3):
        """
        num_categories: number of categories in the dataset, 
        eg. dimension of the one-hot vector
        seq_len: length of the sequence
        size: number of samples in the dataset
        """
        super().__init__()

        assert shift < seq_len, "shift must be less than seq_len"
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size
        self.shift = shift

        # shape: (size, seq_len)
        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = inp_data[self.shift:]
        labels = torch.cat([labels, torch.tensor(inp_data[:self.shift])])
        return inp_data, labels
    

if __name__ == "__main__":
    from functools import partial
    dataset = partial(ShiftedSeqDataset, 10, 16)
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(dataset(1000), batch_size=128)
    test_loader  = data.DataLoader(dataset(10000), batch_size=128)

    inp_data, labels = train_loader.dataset[0]
    print("Input data:", inp_data)
    print("Labels:    ", labels)
