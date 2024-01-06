import torch
import torch.nn as nn

from models.transformer import Transformer 

if __name__ == "__main__":
    print("Hello World!")

    model = Transformer(10000, num_of_decoder_layers=1, num_of_encoder_layers=1) 

    print(model)

    print(model(torch.rand(1, 10, 512), torch.rand(1, 10, 512)).shape)
