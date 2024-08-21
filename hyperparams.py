import torch

block_size = 128
batch_size = 128
dmodel = 256 
learning_rate = 0.00001
cuda_available = torch.cuda.is_available()
dropout = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

finetune_learning_rate = 1e-6
finetune_block_size = 16