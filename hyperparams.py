import torch

block_size = 256
batch_size = 32 
dmodel = 256 
learning_rate = 0.00001
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")