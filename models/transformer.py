import torch
import torch.nn as nn

from .attention import AttentionLayer, FeedForwardLayer 

class PositionalEncoding(nn.Module):
    def __init__(self, block_size, dmodel):
        super(PositionalEncoding, self).__init__()
        self.block_size = block_size
        self.dmodel = dmodel

        encoding = torch.zeros(block_size, dmodel)
        encoding.requires_grad = False

        pos = torch.reshape(torch.arange(0, block_size), (block_size, 1))
        _2i = torch.arange(0, dmodel, step=2)

        encoding[:, 0::2] = torch.sin( pos / 10000**(_2i / dmodel) )
        encoding[:, 1::2] = torch.cos( pos / 10000**(_2i / dmodel) )
        self.encoding = encoding
 
    def forward(self, x):
        # x is a batch of sequences of integers of dim (batch_size, seq_len)
        _, seq_len = x.size()
        return self.encoding[0:seq_len, :]

class Encoder(nn.Module):
    def __init__(self, block_size, dropout=0.1, dmodel=512, num_of_heads=8):
        super(Encoder, self).__init__()
        self.dmodel = dmodel 
        self.block_size = block_size
    
        self.encoder_sa = AttentionLayer(self.block_size, self.dmodel, num_of_heads, False, attention_dropout=dropout)

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
    
        self.decoder_sa = AttentionLayer(self.block_size, self.dmodel, num_of_heads, True, attention_dropout=dropout)

        self.decoder_ca = AttentionLayer(self.block_size, self.dmodel, num_of_heads, False, attention_dropout=dropout)

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
        self.position_encoding = PositionalEncoding(block_size, self.dmodel)

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
        # x: tokenized input
        # y: tokenized previous output
        token_x = x[:, -self.block_size:]
        token_y = y[:, -self.block_size:]

        x = self.embeddings(token_x)
        x += self.position_encoding(token_x).to(x.device)
        y = self.embeddings(token_y)
        y += self.position_encoding(token_y).to(y.device)

        x = self.dropout_encoder_embedding(x)
        y = self.dropout_pre_decoder_embedding(y)

        encoder_output = self.encoder(x)

        for layer in self.decoder:
            y = layer(y, encoder_output)

        output = self.fully_connected(y)

        return output 