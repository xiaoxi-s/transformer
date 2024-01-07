import torch
import torch.nn as nn

from models.transformer import Transformer 

if __name__ == "__main__":
    print("Hello World!")
    block_size = 8
    dmodel = 256 

    model = Transformer(10000, block_size=block_size, dmodel=dmodel) 

    # print(model)
    x = torch.randint(0, 10000, (1, 10), dtype=torch.int64)
    y = torch.randint(0, 10000, (1, 10), dtype=torch.int64)
    print(model(x, y).shape)
