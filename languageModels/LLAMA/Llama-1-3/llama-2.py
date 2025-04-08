from dataclasses import dataclass , field
from typing import Optional, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from inspect import isfunction
"""
NOTE :#The Llama 1 architecture is similar to Llama 2, except that Llama 2 has a larger context window.
Llama 2 has a context window of 4096 tokens, while Llama 1 has a context window of 2048 tokens.
"""
@dataclass 
class ModelConfig:
    dim : int = 4096 
    n_layers : int = 32
    n_heads : int = 32 
    vocab_size : int = 32000 # used senteincepiece tokenizer
    n_kv_heads : int = 32 # used for grouped query attention 
    max_seq_len : int = 4096
    max_batch_size : int = 32
    norm_eps : float = 1e-5
    device : str = field(default_factory=lambda : 'cuda' if torch.cuda.is_available() else 'cpu')


    
#Llamma 2 uses RMSNorm instead of LayerNorm.
#For more details, please see the paper Root Mean Square Layer Normalization (2019)
class RMSNorn(nn.Module):
    def __init__(self , emb_dim , eps = 1e-6):
        pass
class ROPE(nn.Module):
    pass
       
class Llama2(nn.Module):
    pass