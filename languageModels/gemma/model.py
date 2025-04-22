# Gemma from scratch in pytorch 
#Author - Riyansh Shah
#Date - 18 March 2025'

#IMPORTING 
import torch
import torch.nn as nn
import torch.functional as F
import math 
import numpy
from typing import List , Tuple , Dict , Optional
from dataclasses import dataclass , field

@dataclass
class GemmaConfig:
    vocab_size : int
    dim_size : int 
    position_embedding : int
    hidden_layers : int
    num_attention_heads : int
    num_kv_heads : int 
    hidden_size : int
    intermeddiate_size : int
    head_dim : int
    attention_bias : bool = False
    rope_theta : int
    attention_output : bool = True
    pad_token_id : int
    rps_norm_eps : int   
#RMS NORM - ROOT MEAN SQUARE NORMALIZATION 
class RMSNorm(nn.Module):
    def __init__(self , config: GemmaConfig ):
        super().__init__()
        self.dim_size = config.dim_size
        self.eps = config.rps_norm_eps
        self.scale = nn.Parameter(torch.ones(self.dim_size))
        self.shift = nn.Parameter(torch.zeros(self.dim_size))
    def forward(self , x):
        normalize = torch.sqrt(x.pow(2).mean(-1 , keepdim =True) + self.eps)
        return x / normalize
# GATED MULTI-LAYER PERCEPTRON - GEMMA 
class MLP(nn.Module):
    def __init__(self, config : GemmaConfig):
        self.gate_proj = nn.Linear(config.dim_size , config.hidden_size)
        self.up_proj = nn.Linear(config.dim_size , config.hidden_size)
        self.ff3 = nn.Linear(config.hidden_size , config.dim_size)
        self.activation =nn.GELU()
    def forward(self , x ):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated  = self.activation(gate)
        out = activated * up
        output = self.ff3(out)
        return output 
#ROPE - ROTARY POSITIONAL EMBEDDING 
class Rope:
    def __init__():
        pass
# CLASS KV-CACHE DURING INFERENCE 
class KVCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []
    def num_items(self):
        if len(self.key_cache)==0:
            return 0 
        else:
            return self.key_cache[0].shape[-2]
    def update(self , key ,values , layer_index):
        if len(self.key_cache) <= layer_index:
            self.key_cache.append(key)
            self.value_cache.append(values)
        else:
            self.key_cache[layer_index] = torch.cat([self.key_cache[layer_index], key], dim = -2)
            self.value_cache[layer_index] = torch.cat([self.value_cache[layer_index], values], dim = -2)
    def clear(self ):
        self.key_cache.clear()
        self.value_cache.clear()
#REPEAT KEY - VALUE 
def repeat_kv(self , kv , num_repeat):
        batch_size , num_heads , _ ,_ = kv.shape
        if num_repeat ==1:
            return kv 
        else:
            kv = kv[: , : , None , : , :].expand(batch_size , num_heads , num_repeat , -1, -1)
            kv = kv.reshape(batch_size , num_heads * num_repeat , -1 , -1)
            return kv 
#MULTI HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self , config: GemmaConfig):
        super(MultiHeadAttention , self).__init__()
        self.config = config 
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.attention_bias = config.attention_bias
        self.dim = config.dim_size
        self.num_repeat = config.num_attention_heads // config.num_kv_heads
        self.rope_theta = config.rope_theta
        self.w_q = nn.Linear(config.dim_size , config.dim_size , bias = False)
        self.w_k = nn.Linear(config.dim_size , config.dim_size , bias = False)
        self.w_v = nn.Linear(config.dim_size , config.dim_size , bias = False)
        self.w_o = nn.Linear(config.dim_size , config.dim_size , bias = False)
        self.rope = Rope(config.rope_theta)
    def forward(self):
        pass
        