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
class Rope(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.head_dim = config.head_dim                      
        max_seq_len = config.position_embedding              
        theta = float(config.rope_theta)                     
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        sinusoid = torch.einsum("i,j->ij", positions, inv_freq)
        sinusoid = torch.einsum("i,j->ij", positions, inv_freq)
        sin = torch.stack([sinusoid.sin(), sinusoid.sin()], dim=-1).flatten(1)
        cos = torch.stack([sinusoid.cos(), sinusoid.cos()], dim=-1).flatten(1)
        self.register_buffer("sin", sin, persistent=False)   
        self.register_buffer("cos", cos, persistent=False)
    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape [batch_size, seq_len, num_heads, head_dim]
        returns: same shape, with rotary embeddings applied to the last dim
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        assert head_dim == self.head_dim, "Head dim mismatch"
        sin = self.sin[:seq_len]   .unsqueeze(0).unsqueeze(2)  
        cos = self.cos[:seq_len]   .unsqueeze(0).unsqueeze(2)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        x_rot = torch.cat((-x_odd.unsqueeze(-1), x_even.unsqueeze(-1)), dim=-1).view_as(x)
        return x * cos + x_rot * sin
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
    def __init__(self , config: GemmaConfig , cache : KVCache):
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
        self.repeat_kv = self.repeat_kv
    def forward(self , x , cache : KVCache , layer_index , attention_mask = None ):
        batch_size , seq_len , _ = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch_size , seq_len , self.num_attention_heads , self.head_dim)
        k = k.view(batch_size , seq_len , self.num_kv_heads , self.head_dim)
        v = v.view(batch_size , seq_len , self.num_kv_heads , self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        k = self.repeat_kv(k , self.num_repeat)
        v = self.repeat_kv(v , self.num_repeat)
        if cache is not None: 
            k , v = cache.update(k , v , layer_index)
        attention_weights = torch.matmul(q , k.transpose(-1 , -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights , dim = -1)
        attention_output = torch.matmul(attention_weights , v)
        attention_output = attention_output.transpose(-1 , -2)
        attention_output = attention_output.contiguous().view(batch_size , seq_len , self.num_attention_heads , self.head_dim)
        attention_output = self.w_o(attention_output)
        return attention_output , attention_weights
#DECODER LAYER 
class DecoderLayer(nn.Module):
    def __init__(self , config : GemmaConfig):
        super(DecoderLayer , self).__init__()
        self.config = config
        self.attention = MultiHeadAttention(config , cache = KVCache())
        self.mlp = MLP(config)
        self.rms1 = RMSNorm(config)
        self.rms2 = RMSNorm(config)
    def forward(self , x , layer_id , attention_mask = None ):
        residual = x 
        x = self.rms1(x) #PreNorm in GEMMA
        x , _ = self.attention(x , attention_mask = attention_mask , layer_index = layer_id)
        x = residual + x #Residual connection
        x = self.rms2(x) #PostNorm in GEMMA
        x = self.mlp(x)
        x = residual + x #Residual connection
        return x
#GEMMA MODEL 
class Gemma(nn.Module):
    def __init__(self , config : GemmaConfig):
        super(Gemma , self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size , config.dim_size)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.hidden_layers)])
        self.layernorm = RMSNorm(config)
        self.output = nn.Linear(config.dim_size , config.vocab_size)
    def forward(self, x , attention_mask : bool = True , cache : Optional[KVCache] = None):
        pass #TODO