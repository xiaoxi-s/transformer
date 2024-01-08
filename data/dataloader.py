import torch

from torch.utils.data import Dataset 

class BabyShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]