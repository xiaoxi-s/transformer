import torch
import torch.nn as nn

from .attention import AttentionLayer, FeedForwardLayer 


class Encoder(nn.Module):
    def __init__(self, block_size, dropout=0.1, dmodel=512, num_of_heads=8):
        super(Encoder, self).__init__()
        self.dmodel = dmodel 
        self.block_size = block_size
    
        self.encoder_sa = AttentionLayer(self.dmodel, num_of_heads, False, attention_dropout=dropout)

        self.encoder_ffwd = FeedForwardLayer(self.dmodel, dropout=dropout) 

    def forward(self, x):
        x = self.encoder_sa(x, x, x)
        x = self.encoder_ffwd(x)
        return x

class Decoder(nn.Module):
    def __init__(self, block_size, dropout=0.1, dmodel=512, num_of_heads=8):
        super(Decoder, self).__init__()
        self.dmodel = dmodel 
        self.block_size = block_size
    
        self.decoder_sa = AttentionLayer(self.dmodel, num_of_heads, True, attention_dropout=dropout)

        self.decoder_ca = AttentionLayer(self.dmodel, num_of_heads, False, attention_dropout=dropout)

        self.decoder_ffwd = FeedForwardLayer(self.dmodel, dropout=dropout)
    
    def forward(self, y, encoder_output):
        y = self.decoder_sa(y, y, y)
        y = self.decoder_ca(y, encoder_output, encoder_output)
        y = self.decoder_ffwd(y)
        return y

class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, dropout=0.1, dmodel=512, num_of_encoder_layers=6, num_of_decoder_layers=6, num_of_heads=8):
        super(Transformer, self).__init__()
        self.dmodel = dmodel 
        self.block_size = block_size

        self.embeddings = nn.Embedding(vocab_size, self.dmodel)
        self.position_embedding_table = nn.Embedding(block_size, self.dmodel)

        # hard coded dropout
        self.dropout_encoder_embedding = nn.Dropout(0.1)
        self.dropout_pre_decoder_embedding = nn.Dropout(0.1)

        self.encoder = nn.Sequential(
            *[Encoder(self.block_size, dropout=dropout, dmodel=self.dmodel, num_of_heads=num_of_heads) for _ in range(num_of_encoder_layers)]
        )

        self.decoder = nn.ModuleList(
            [Decoder(self.block_size, dropout=dropout, dmodel=self.dmodel, num_of_heads=num_of_heads) for _ in range(num_of_decoder_layers)]
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.dmodel, vocab_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialize weights according to https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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

        encoder_output = self.encoder(x)

        for layer in self.decoder:
            y = layer(y, encoder_output)

        output = self.fully_connected(y)

        return output 