import torch
import torch.nn as nn

# dq, dk, dv, dmodel = 64, 64, 64, 512

class Attention(nn.Module):
    def __init__(self, dmodel=512, dk=64, mask=False):
        super(Attention, self).__init__()
        self.dmodel = dmodel 
        self.dk = dk
        self.mask = mask
    
    def forward(self, Q, K, V):
        # Q: (batch_size, seq_len, dq)
        # K: (batch_size, seq_len, dk)
        # V: (batch_size, seq_len, dv)
        # print(Q.shape, K.shape, V.shape)

        output = Q @ K.mT / (self.dk**0.5)
        if self.mask:
            mask = torch.triu(torch.ones(output.shape), diagonal=1)
            mask.masked_fill_(mask==1, float('-inf'))
        else:
            mask = torch.zeros(output.shape)

        output = torch.softmax((output + mask), dim=-1)
        return output @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel, num_heads, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.dmodel = dmodel 
        self.num_heads = num_heads
        self.dk = dmodel // num_heads
        self.dv = self.dk

        self.WQ = nn.Linear(dmodel, self.dk)
        self.WK = nn.Linear(dmodel, self.dk)
        self.WV = nn.Linear(dmodel, self.dv)

        self.heads = nn.ModuleDict({
            f'head_{i}': Attention(self.dmodel, self.dk, mask)
            for i in range(self.num_heads)
        })

        self.linear = nn.Linear(self.num_heads * self.dv, self.dmodel)

    def forward(self, Q, K, V):
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)

        # print("heads output shape: ", self.heads['head_0'](Q, K, V).shape)

        output = torch.cat([head(Q, K, V) for head in self.heads.values()], dim=-1)
        output = self.linear(output)

        return output 


class AttentionLayer(nn.Module):
    def __init__(self, dmodel, num_heads, mask=False, attention_dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.dmodel = dmodel 
        self.num_heads = num_heads
        self.mask = mask

        self.attention = MultiHeadAttention(self.dmodel, self.num_heads, self.mask)
        self.layer_norm = nn.LayerNorm(self.dmodel)
        self.dropout_1 = nn.Dropout(attention_dropout)

        self.fully_connected = nn.Sequential(
            nn.Linear(self.dmodel, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.dmodel)
        )
        self.dropout_2 = nn.Dropout(attention_dropout)
        self.layer_norm_2 = nn.LayerNorm(self.dmodel)

    def forward(self, Q, K, V):
        # Q, K, V: (batch_size, seq_len, dmodel)
        x = V 
        output = self.attention(Q, K, V)
        output = self.dropout_1(output)
        x = self.layer_norm(output + x)
        output = self.fully_connected(output)
        output = self.dropout_2(output)
        output = self.layer_norm_2(output + x)

        return output