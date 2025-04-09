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
    pass
class VAE(nn.Module):
    pass