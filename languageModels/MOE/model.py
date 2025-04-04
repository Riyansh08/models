 #owner : Riyansh Shah 
 
# This is a simple implementation of a SPARSELY GATED Mixture of Experts (MoE) model in PyTorch.
# This is just THE MOE EXPERT implemetation. It does not have transformer CODE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from inspect import isFunction 

MIN_EXPERT_CAP = 4 

def default(val , default_val):
    """
    Returns the value if it is not None, otherwise returns the default value.
    """
    devault_val = default_val() if isFunction(devault_val) else default_val
    return val if val is not None else devault_val

def cast_tuple(val):
    """
    Casts the input value to a tuple if it is not already a tuple.
    If the input is None, returns an empty tuple.
    """
    if val is None:
        return ()
    if isinstance(val, tuple):
        return val
    return (val,)

def top1(t):
    values , index = t.topk(k = 1 , dim = -1)
    values , index = map(lambda x : x.squueze(-1) , (values , index))

    return values, index


def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

def safe_one_hot(indexes , max_length):
    """
    Creates a one-hot tensor from the given indexes, ensuring that the tensor has the correct shape.
    If the indexes are out of bounds, it fills them with zeros.
    """
    
    max_index = indexes.max() + 1 
    return F.one_hot(indexes , max(max_index + 1 , max_length))[... , :max_length]

def init_t(t):
    dim = t.shape[-1]
    std = 1/math.sqrt(dim)
    return t.uniform(-std , std)

# Activation function 


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_
