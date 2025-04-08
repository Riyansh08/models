# Note - GPT-2 Implementation for reference 

from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np

GPT2_CONFIG_124M_PARAMETERS = {
    'vocab_size': 50257,
    'context_size' : 256,
    'n_emb' : 768,
    'n_layer' : 12,
    'n_head' : 12,
    'n_inner_mlp ': 3072,
    'atn_drop': 0.1,
    'att_bias' : False
} 

class MultiHeadAttention(nn.Module):
    def __init__(self , config ):
        super(MultiHeadAttention , self).__init__()
        assert config['n_emb' ] % config['n_head'] == 0  , "n_emb should be divisible by n_head"
        
        self.n_emb = config['n_emb']
        self.n_head = config['n_head']
        self.n_emb_per_head = self.n_emb // self.n_head
        self.w_q = nn.Linear(self.n_emb , self.n_emb , bias = config['att_bias'])
        self.w_k = nn.Linear(self.n_emb , self.n_emb , bias = config['att_bias'])
        self.w_v = nn.Linear(self.n_emb , self.n_emb , bias = config['att_bias'])
        
        self.w_o = nn.Linear(self.n_emb , self.n_emb , bias = config['att_bias'])
        self.dropout = nn.Dropout(config['atn_drop'])
        self.register_buffer("mask" , torch.tril(torch.ones(config['context_size'] , config['context_size'])))
        
    def forward(self , x ):
        B , T ,C = x.shape 
        query = self.w_q(x).view(B , T , self.n_head , self.n_emb_per_head).transpose(1 , 2)
        key = self.w_k(x).view(B , T , self.n_head , self.n_emb_per_head).transpose(1 , 2)
        value = self.w_v(x).view(B , T , self.n_head , self.n_emb_per_head).transpose(1 , 2)
        attention_scores = torch.matmul(query , key.transpose(-2 , -1))
        mask_bool = self.mask.bool()[:T , :T]
        attention_scores.masked_fill_(mask_bool , float['-inf'])
        attention_weights = torch.softmax(attention_scores / math.sqrt(self.n_emb_per_head) , dim = -1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights , value)
        attention_output = attention_output.transpose(1 , 2).contiguous().view(B , T , C)
        final_output = self.w_o(attention_output)
        return final_output , attention_weights
class LayerNormalization(nn.Module):
    def __init__(self , n_emb):
        super(LayerNormalization , self).__init__()
        self.scale =nn.Parameter(torch.ones(n_emb))
        self.bias = nn.Parameter(torch.zeros(n_emb))
        self.eps = 1e-4 #original 1e-5
        
    def forward(self , x):
        mean = x.mean(dim = -1  ,keepdim = True)
        var = x.var(dim = -1 , keepdim = True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x  = self.scale * x +self.bias 
        return x 
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU , self).__init__()
        
    def forward(self , x):
        return x * 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x , 3))))
    
class MLP(nn.Module):
    def __init__(self , config):
        super(MLP , self).__init__()
        self.fc1 = nn.Linear(config['n_emb'] , config['n_inner_mlp'] , bias = True)
        self.activation = GeLU()
        self.fc2 = nn.Linear(config['n_inner_mlp'] , config['n_emb'] , bias = True)
        
    def forward(self , x):
        x = self.fc2(self.activation(self.fc1(x)))
        return x 
class Block(nn.Module):
    def __init__(self , config):
        super(Block , self).__init__()
        self.attention = MultiHeadAttention(config)
        self.layerNorm = LayerNormalization(config['n_emb'])
        self.mlp = MLP(config)
        self.layerNorm2 = LayerNormalization(config['n_emb'])
        self.dropout = nn.Dropout(config['atn_drop'])
        
    def forward(self , x ):
        residual = x 
        attention_output , _ = self.attention(x)
        x  = self.layerNorm(residual + attention_output)
        
        residual = x 
        mlp_output = self.mlp(x)
        x = self.layerNorm2(residual + mlp_output)
        return x 
    
class PositionalEmbedding(nn.Module):
    def __init__(self , config , x):
        super(PositionalEmbedding , self).__init__()
        self.n_emb = config['n_emb']
        self.context_size = config['context_size']
    
        pe = torch.zeros(self.context_size , self.n_emb)
        divid_term = torch.exp(torch.arange(0 , self.n_emb , 2) * -(math.log(10000.0) / self.n_emb))
        pe[:, 0::2] = torch.sin(x[:, 0::2] * divid_term)
        pe[:, 1::2] = torch.cos(x[:, 1::2] * divid_term)
        self.register_buffer('pe', pe) 
        
    def forward(self , x):
         seq_len = x.size(1)
         return self.pe[:seq_len].unsqueeze(0)  # shape: [1, seq_len, n_emb]
        
class GPT2(nn.Module):
    def __init__(self , config):
        super(GPT2 , self).__init__()
        self.n_emb = config['n_emb']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.vocab_size = config['vocab_size']
        self.context_size = config['context_size']
        
        self.emb = nn.Embedding(self.vocab_size , self.n_emb)
        self.positional_emb = PositionalEmbedding(config)
        self.drop = nn.Dropout(config['atn_drop'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.n_layer)])
        self.final_norm = LayerNormalization(self.n_emb)
        self.out = nn.Linear(self.n_emb , self.vocab_size , bias = False
        )
    def forward(self , x , ):
        B , T  = x.shape
        assert T <= self.context_size , "Input sequence length exceeds context size"    
        tokens = self.emb(x)
        position = self.positional_emb(self.config , x)
        x = self.drop(tokens + position) 
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits 