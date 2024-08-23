import torch

block_size = 128
batch_size = 1024
dmodel = 256 
num_of_decoder_layers = 4
num_of_encoder_layers = 4
learning_rate = 1e-6
cuda_available = torch.cuda.is_available()
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

finetune_learning_rate = 1e-6
finetune_block_size = 3
finetune_batch_size = 16