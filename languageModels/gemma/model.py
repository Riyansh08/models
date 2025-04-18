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
#RMS NORM - ROOT MEAN SQUARE NORMALIZATION 
class RMSNorm(nn.Module):
    def __init__(self , config: GemmaConfig ):
        super().__init__()
        self.dim_size = config.dim_size
        
    

#ROPE - ROTARY POSITIONAL EMBEDDING 
class Rope:
    def __init__():
        pass
