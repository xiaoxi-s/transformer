import torch

from collections.abc import Iterable
from torch.utils.data import Dataset 

class BabyShakespeareDataset(Dataset):
    def __init__(self, data, block_size, device):
        # self.data = data.to(device)
        self.data = data
        self.block_size = block_size
        self.device = device

    def __len__(self):
        # -1 for the last token won't have a next token
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]