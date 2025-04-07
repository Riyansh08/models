import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from inspect import isfunction
"""
NOTE :#The Llama 1 architecture is similar to Llama 2, except that Llama 2 has a larger context window.
Llama 2 has a context window of 4096 tokens, while Llama 1 has a context window of 2048 tokens.
"""

#Llamma 2 uses RMSNorm instead of LayerNorm.
#For more details, please see the paper Root Mean Square Layer Normalization (2019)
class RMSNorn(nn.Module):
    def __init__(self , emb_dim , eps = 1e-6):
        pass
       
class Llama2(nn.Module):
    pass