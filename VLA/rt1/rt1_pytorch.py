# RT1 Pytorch 
# This code implements a RT1 from scratch using PyTorch.
# Date - 11 May 2025
# Reference - https://github.com/lucidrains/robotic-transformer-pytorch/tree/main/robotic_transformer_pytorch
# Thankyou Lucidrains for open-source contributions 

import torch 
from __future__ import annotations 
from torch.nn import Module , ModuleList 
import torch.nn.functional as F 
import math 
import numpy as np 
from torch import nn , einsum , Tensor 
from einops import pack, unpack, repeat, reduce, rearrange
from typing import Callable
from beartype import beartype
from einops.layers.torch import Rearrange, Reduce
from functools import partial
from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner , classifier_free_guidance

def exists(val):
    return val is not None 

#TO-DO 