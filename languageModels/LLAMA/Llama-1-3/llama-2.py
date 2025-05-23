#lLama 2 
from dataclasses import dataclass , field
from typing import Optional, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from inspect import isfunction
"""
NOTE : The Llama 1 architecture is almost similar to Llama 2, except that Llama 2 has a larger context window and other minor changes .
Llama 2 has a context window of 4096 tokens, while Llama 1 has a context window of 2048 tokens.
"""
@dataclass 
class Config:
    dim : int = 4096 
    hidden_dim : int = 11008
    ffn_dim : int = 11008
    n_layers : int = 32
    n_heads : int = 32 
    kv_heads : int = 16
    vocab_size : int = 32000 # used senteincepiece tokenizer
    n_kv_heads : int = 32 # used for grouped query attention 
    max_seq_len : int = 4096
    max_batch_size : int = 32
    norm_eps : float = 1e-5
    device : str = field(default_factory=lambda : 'cuda' if torch.cuda.is_available() else 'cpu')

#Llamma 2 uses RMSNorm instead of LayerNorm.
#For more details, please see the paper Root Mean Square Layer Normalization (2019)
class RMSNorm(nn.Module):
    def __init__(self , emb_dim , eps = 1e-6):
        super(RMSNorm , self).__init__()
        self.eps = eps
        self.dim = emb_dim
        self.scale = nn.Parameter(torch.ones(emb_dim))
        
    def forward(self , x ):
        means = x.pow(2).mean(dim = -1 , keepdim = True)
        x_normalized = x * torch.rqrt(means + self.eps)
        x_normalized = x_normalized * self.scale
#Activation Function 
#Llama 2 uses the SiLU activation function instead of GELU.
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU , self).__init__()
    def forward(self , x ):
        return x * 1/(1 +torch.exp(-x))   
class SwiGLU_FFN(nn.Module):
    def __init__(self , config : Config):
        super(SwiGLU_FFN , self).__init__()
        self.config = config
        self.ffn1 = nn.Linear(config.dim , config.ffn_dim)
        self.ffn2 = nn.Linear(config.dim , config.ffn_dim)
        self.ffn3 = nn.Linear(config.ffn_dim , config.dim)
        self.activation = SiLU()
    def forward(self , x):
        x_fc1 = self.ffn1(x)
        x_fc2 = self.ffn2(x)
        x = self.activation(x_fc1) * x_fc2
        x = self.ffn3(x)
        return x  
class ROPE(nn.Module):
    def __init__(self , h_dim , seq_len):
        super(ROPE , self).__init__()
        self.h_dim = h_dim
        self.seq_len = seq_len 
        assert self.h_dim % 2 == 0 , "head dimensions must be even "
        theta_numerator = torch.arrange(0 , self.h_dim ,2 , dtype = torch.float32)
        theta = 1.0 / torch.pow(10000 , theta_numerator / self.h_dim)
        m = torch.arange(self.seq_len , dtype = torch.float32).unsqueeze(1)       
        freqs = torch.matmul(m , theta.unsqueeze(0))
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    def forward(self, x, start_pos) :
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim, "dim must be equal to self.dim"
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freq_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freq_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(x.device)    
class MultiHeadAttention(nn.Module):
    def __init__(self , config : Config , multiquery = False):
        super(MultiHeadAttention , self).__init__()
        self.config = config
        assert config.dim % config.n_heads == 0 , "dimension must be divisible by number of heads"
        self.head_dim  = config.dim // config.n_heads 
        self.q_n_heads = config.n_heads
        self.kv_heads = config.kv_heads if multiquery else config.n_heads
        self.n_rep = config.n_heads // self.kv_heads #for groupquery attention
        self.q_w = nn.Linear(config.dim , config.dim , bias = False)
        self.k_w = nn.Linear(config.dim , config.dim , bias = False)
        self.v_w = nn.Linear(config.dim , config.dim , bias = False)
        self.o_w = nn.Linear(config.dim , config.dim , bias = False)
        #KEY-VALUE CACHE FOR INFERENCE 
        self.key_cache = torch.zeros((config.max_batch_size , config.max_seq_len , self.kv_heads , self.head_dim))
        self.value_cache = torch.zeros((config.max_batch_size , config.max_seq_len , self.kv_heads , self.head_dim))
        #ROPE - ROTARY POSITION EMBEDDINGS
        self.rope = ROPE(self.head_dim , config.max_seq_len)
    @staticmethod
    def group_query_attention(self , config , x ):
        n_rep = config.n_heads // config.n_kv_heads
        batch_size , seq_len , n_kv_heads , head_dim = x.shape
        if n_rep == 1:
            return x 
        else:
            return (x[: , : , : , None , :]
                    .expand(batch_size , seq_len , n_kv_heads , n_rep , head_dim)).reshape(batch_size , seq_len , n_kv_heads * self.n_rep , head_dim)
    def forward(self, x ,config  , start_pos = 0):
        batch_size , seq_len , dim = x.shape  
        assert dim == self.config.dim , "dimension must be equal to config.dim"              
        query = self.q_w(x)
        value = self.v_w(x)
        key = self.k_w(x)
        query  = query.view(batch_size , seq_len , self.q_n_heads , self.head_dim)
        #KEY AND VALUE HEADS- IF GROUPEDQUERY THEN USE KV HEADS
        value = value.view(batch_size , seq_len , self.kv_heads , self.head_dim)
        key = key.view(batch_size , seq_len , self.kv_heads , self.head_dim)       
        #APPLY ROPE 
        query = self.rope(query , start_pos = 0)
        key = self.rope(key , start_pos = 0)        
        #KEY-VALUE CACHE RETRIEVAL
        self.key_cache[:batch_size , start_pos:start_pos + seq_len , : , :] = key
        self.value_cache[:batch_size , start_pos:start_pos + seq_len , : , :] = value
        keys = self.key_cache[:batch_size , :start_pos + seq_len , : , :] 
        values = self.value_cache[:batch_size , :start_pos + seq_len , : , :]
        #REPEAT THE KEY AND VALUE HEADS
        keys = self.group_query_attention(config , keys)
        values = self.group_query_attention(config , values)
        #CRUX OF THE ATTENTION MECHANISM
        query = query.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        scores = torch.matmul(query , keys.transpose(-2,-1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores , dim = -1)
        attention_scores = torch.matmul(scores , values)
        attention_scores = attention_scores.transpose(1,2).contiguous().view(batch_size , seq_len , dim)
        output = self.o_w(attention_scores)
        return output       
class DecoderBlocks(nn.Module):
    def __init__(self , config: Config):
        super(DecoderBlocks , self).__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.ffn = SwiGLU_FFN(config)
        self.norm1 = RMSNorm(config.dim , eps = config.norm_eps)
        self.norm2 = RMSNorm(config.dim , eps = config.norm_eps)
    def forward(self , x , config , start_pos = 0):
        out = x + self.attention(self.norm1(x) , config , start_pos = start_pos) 
        final = out + self.ffn(self.norm2(out))
        return final       
class Llama2(nn.Module):
    def __init__(self , config : Config):
        super(Llama2 , self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size , config.dim )
        self.blocks = nn.ModuleList([DecoderBlocks(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim , eps = config.norm_eps)
        self.output = nn.Linear(config.dim , config.vocab_size, bias = False)
    def forward(self , x , start_pos = 0):
        batch_size , seq_len = x.shape
        assert seq_len <= self.config.max_seq_len , "sequence length must be less than max sequence length"
        assert batch_size <= self.config.max_batch_size , "batch size must be less than max batch size"
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x , self.config , start_pos = start_pos)
        x = self.norm(x)
        x = self.output(x)
        return x
        