import torch
import torch.nn as nn

from .attention import AttentionLayer 

class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, dropout=0.3, dmodel=512, num_of_encoder_layers=6, num_of_decoder_layers=6, num_of_heads=8):
        super(Transformer, self).__init__()
        self.dmodel = dmodel 
        self.block_size = block_size

        self.embeddings = nn.Embedding(vocab_size, self.dmodel)
        self.position_embedding_table = nn.Embedding(block_size, self.dmodel)
        self.dropout_encoder_embedding = nn.Dropout(dropout)
        self.dropout_pre_decoder_embedding = nn.Dropout(dropout)

        self.encoder = nn.ModuleDict({
            f"encoder_layer_{i}": AttentionLayer(self.dmodel, num_of_heads, False, attention_dropout=dropout)
            for i in range(num_of_encoder_layers)
        })

        self.pre_decoder = nn.ModuleDict({
            f"pre_decoder_layer_{i}": AttentionLayer(self.dmodel, num_of_heads, True, attention_dropout=dropout)
            for i in range(num_of_decoder_layers)
        })

        self.decoder = nn.ModuleDict({
            f"decoder_layer_{i}": AttentionLayer(self.dmodel, num_of_heads, False, attention_dropout=dropout)
            for i in range(num_of_decoder_layers)
        })

        self.fully_connected = nn.Sequential(
            nn.Linear(self.dmodel * num_of_decoder_layers, vocab_size)
        )
    
    def forward(self, x, y):
        # x: embeddings of input
        # y: embeddings of previous output
        # print(type(x))
        # print(x.shape)
        x = x[:, -self.block_size:]
        y = y[:, -self.block_size:]

        x = self.embeddings(x) + self.position_embedding_table(torch.arange(x.shape[1]))
        y = self.embeddings(y) + self.position_embedding_table(torch.arange(y.shape[1]))

        x = self.dropout_encoder_embedding(x)
        y = self.dropout_pre_decoder_embedding(y)

        # encodings = torch.cat([self.encoder[f"encoder_layer_{i}"](x, x, x) for i in range(len(self.encoder))], dim=-1)
        # pre_decodings = torch.cat([self.pre_decoder[f"pre_decoder_layer_{i}"](y, y, y) for i in range(len(self.pre_decoder))], dim=-1)
        encodings = [self.encoder[f"encoder_layer_{i}"](x, x, x) for i in range(len(self.encoder))]
        pre_decodings = [self.pre_decoder[f"pre_decoder_layer_{i}"](y, y, y) for i in range(len(self.pre_decoder))]

        # print("shapes: ", len(pre_decodings), len(encodings))
        output = torch.cat([self.decoder[f"decoder_layer_{i}"](pre_decodings[i], encodings[i], encodings[i]) for i in range(len(self.decoder))], dim=-1)

        output = self.fully_connected(output)

        return output 