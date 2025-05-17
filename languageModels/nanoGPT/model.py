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
    n_embd:int = 1600
    norm_eps: float = 1e-5
    
    
    
# evaluate the base gpt2
# n_layer=48, n_head=25, n_embd=1600
# 1558M parameters
# batch_size = 8
# eval_iters = 500 # use more iterations to get good estimate
# eval_only = True
# wandb_log = False
# init_from = 'gpt2-xl'
    
#LayerNormalization 

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias"""
    def __init__(self , config : NanoGPTConfig):
        super(LayerNorm , self).__init__()
        