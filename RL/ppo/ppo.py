import torch
import torch.nn as nn
import torch.nn.functional as F
import math , numpy as np  , random  
from typing import List , Tuple , Dict , Optional 
from dataclasses import dataclass , field
from torch.distributions import MultivariateNormal , Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

#SETTING DEVICE 
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print("USING A GPU ")
else:
    print("USING A CPU ")
#PPO POLICY 
class RoloutBuffer():
    def __init__(self):
        pass