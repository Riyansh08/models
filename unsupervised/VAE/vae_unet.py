#VAE - using U-Net architecture
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init as init
import numpy as np 
from dataclasses import dataclass, field , asdict
from typing import List, Optional, Tuple
# NOTE - output size - [input - Kernel + 2*padding] / stride + 1
@dataclass
class Config:
    """
    Configuration class for the VAE model.
    """
    batch_size : int = 64
    num_epochs : int 
    learning_rate : float = 0.01 
    x_dim : int 
    z_dim : int
    hidden_dim1 : int
    hidden_dim2 : int
    hidden_dim3 : int
    dropout : float = 0.2
    in_channels : int
    out_channelss : int
    kernel_size :int
    stride : int
    padding : int 
#CNN from Scratch 
class Convolutional:
    def __init__(self ):
        pass
class Encoder(nn.Module):
    pass
class VAE(nn.Module):
    pass