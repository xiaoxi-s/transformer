import torch

block_size = 256
batch_size = 256
dmodel = 256 
learning_rate = 0.00001
cuda_available = torch.cuda.is_available()
dropout = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_decoder_layers=4
num_of_encoder_layers=4

# Not used for now
# finetune_learning_rate = 1e-6
# finetune_block_size = 3
# finetune_batch_size = 16