# NanoGPT MODEL WITH Flash Attention 
import torch
import torch.nn as nn  
import math  
import inspect 
import numpy as np
from torch.nn import functional as F 
from dataclasses import dataclass 
import wandb # Weight and Bias 

@dataclass
class NanoGPTConfig:
    batch_size : int = 8
    n_embd:int = 1600
    norm_eps: float = 1e-5
    norm_bias : bool = True
    attention_bias : bool = False 
    batch_norm_momentum : float = 0.999
    n_layer : int  = 48
    n_head : int = 25
    vocab_size : int = 50257
    block_size : int = 1024 # Context Length
    dropout : float = 0.01
    device : str = 'cuda' if torch.cuda.is_available else 'cpu'
    
# 1558M parameters

#LayerNormalization 

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias"""
    def __init__(self , config : NanoGPTConfig):
        super(LayerNorm , self).__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))
        self.eps = config.norm_eps 
    def forward(self , x):
        mean = x.mean(dim= -1 , keepdim = True)
        variance = x.var(dim = -1 , keepdim = True)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return ( self.weight * x_norm ) + self.bias if self.config.norm_bias else self.weight * x_norm 
#Used Earlier using Running Mean and variance
class BatchNorm(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(BatchNorm , self).__init__()
        self.config = config 
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))
        self.eps = config.norm_eps 
        self.register_buffer('running_mean' , torch.zeros(config.n_embd))
        self.register_buffer('running_variance' , torch.ones(config.n_embd))
        self.momentum = config.batch_norm_momentum
    def forward(self , x):
        if self.training:
          batch_mean = x.mean(dim = 0 , keepdim = True)
          batch_variance = x.var(dim = 0 , keepdim = True)
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
          self.running_variance = (1 - self.momentum) * self. running_variance + self.momentum * batch_variance
          x_norm = (x - batch_mean) / torch.sqrt(batch_variance + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_variance + self.eps)
        return (self.weight * x_norm + self.bias) if self.config.norm_bias else self.weight * x_norm

#Activation Function 
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU , self).__init__()
    def forward(self , x):
        return (0.5 * x ) * ( 1.0 + torch.erf(x / math.sqrt(2.0)))

class SwiGeLU(nn.Module):
    def __init__(self):
        super(SwiGeLU , self).__init__()
    def forward(self , x ):
        return x * torch.sigmoid(1.702 * x)
#Multi - Layer - Perceptron 

class MLP(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(MLP , self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.n_embd , 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd  , config.n_embd)
        self.act = SwiGeLU()
    def forward(self , x):
        return self.fc2(self.act(self.fc1(x)))
#Casual Self-Attention 
# Flash Attention 
class CasualSelfAttention(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(CasualSelfAttention , self).__init__()
        self.config = config 
        assert config.n_embd % config.n_head == 0 , "dimensions must be divisible by number of heads"
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.attention_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout 
        # QKV Projections 
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd  , bias = config.attention_bias)
        self.c_proj = nn.Linear(config.n_embd , config.n_embd , bias = config.attention_bias)
        # Check if Flash Attention is available
        self.flash = hasattr(torch.nn.functional , 'scaled_dot_product_attention')
        if not self.flash:
            print('WARNING : Flash attention is not available , using slow attention ')
            self.register_buffer('casual_mask' , torch.tril(torch.ones(config.block_size , config.block_size)).view(1 , 1 , config.block_size , config.block_size))
        #Forward 
        def forward(self , x ): 
            B , T , C = x.size()
            q , k , v = self.c_attn(x).split(self.n_embd , dim = -1)
            k = k.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 )
            v = v.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 ) # B , n_head  , T , C // n_head
            q = q.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 ) #  B , n_head  , T , C // n_head
            
            if self.flash:
                out = torch.nn.functional.scaled_dot_product_attention(q , k , v , attn_mask = None , dropout = self.dropout if self.training else 0.0 , is_casual = True)
            else:
                #Slow Attention 
                # k becomes B , n_head , C // n_head, T
                att = (q @ k.transpose(-2 , -1 )) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.casual_mask[: , : , :T , :T ] == 0 , float('-inf'))
                att = F.softmax(att , dim = -1)
                att = self.attention_dropout(att)
                # B , n_head , T , T @ B , n_head , T ,C //n_head -> B  , n_head , T , C// n_head
                out = att @ v
            
            out = out.transpose(1 , 2 ).contiguous().view(B , T , C)
            out = self.c_proj(out)
            return out
            print("Done")
        
       
class Block(nn.Module):
    def __init__(self , config):      
            super(Block , self).__init__()
            self.ln_1 = LayerNorm(config)
            self.attention = CasualSelfAttention(config)
            self.ln_2 = LayerNorm(config)
            self.mlp = MLP(config)
    def forward(self , x ):
        # Pre-Normalization
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class NanoGPT(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(NanoGPT , self).__init__()
        self.config = config
        assert config.vocab_size is not None , "vocab_size is required"
        assert config.block_size is not None , "Context Length is required"
        pass
        
        
        
                
