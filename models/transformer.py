import torch
import torch.nn as nn

from .attention import AttentionLayer 

class Transformer(nn.Module):
    def __init__(self, vocab_size, dmodel=512, num_of_encoder_layers=6, num_of_decoder_layers=6, num_of_heads=8):
        super(Transformer, self).__init__()
        self.dmodel = dmodel 

        self.encoder = nn.ModuleDict({
            f"encoder_layer_{i}": AttentionLayer(self.dmodel, num_of_heads, False)
            for i in range(num_of_encoder_layers)
        })

        self.pre_decoder = nn.ModuleDict({
            f"pre_decoder_layer_{i}": AttentionLayer(self.dmodel, num_of_heads, True)
            for i in range(num_of_decoder_layers)
        })

        self.decoder = nn.ModuleDict({
            f"decoder_layer_{i}": AttentionLayer(self.dmodel, num_of_heads, False)
            for i in range(num_of_decoder_layers)
        })

        self.fully_connected = nn.Sequential(
            nn.Linear(self.dmodel * num_of_decoder_layers, vocab_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x, y):
        # x: embeddings of input
        # y: embeddings of previous output

        encodings = torch.cat([self.encoder[f"encoder_layer_{i}"](x, x, x) for i in range(len(self.encoder))], dim=-1)
        pre_decodings = torch.cat([self.pre_decoder[f"pre_decoder_layer_{i}"](y, y, y) for i in range(len(self.pre_decoder))], dim=-1)
        output = torch.cat([self.decoder[f"decoder_layer_{i}"](pre_decodings[i], encodings[i], encodings[i]) for i in range(len(self.decoder))], dim=-1)

        output = self.fully_connected(output)

        return output 