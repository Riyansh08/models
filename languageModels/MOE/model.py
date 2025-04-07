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

#Initialization function - Xavier Like Initializatio
 
def init_t(t):
    dim = t.shape[-1]
    std = 1/math.sqrt(dim)
    return t.uniform(-std , std)

# Activation function -  Gaussian Error Linear Unit.

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class Swish(nn.Moudule):
    def forward(self, x):
        return x * torch.sigmoid(x)
  
# Main Implementation 
#expert class 

class MoeExperts(nn.Module):
    def __init__(self , 
                 dim , 
                 num_experts = 16, 
                 hidden_dim = None, 
                 activation = GELU, 
                 init_method = init_t):
        super(MoeExperts , self).__init__()
        
        hidden_dim = default(hidden_dim , dim * 4)
        num_experts = cast_tuple(num_experts)
        w1 = torch.zeros(num_experts , dim , hidden_dim)
        w2 = torch.zeros(num_experts , hidden_dim , dim )
        w1 = init_method(w1)
        w2 = init_method(w2)
        self.weight1 = nn.Parameter(w1)
        self.weight2 = nn.Parameter(w2)
        
        self.activation = activation 
    def forward(self , x):
        hidden = torch.einsum('...nd,...dh->...nh' , x , self.weight1)
        hidden = self.activation(hidden)
        output = torch.einsum('...nh,...hd->...nd' , hidden , self.weight2)
        return output
        # x = torch.einsum('...nd,...dh->...nh' , x , self.weight1)       
# Top k gating function 
class TopKGate(nn.Module):
    def __init__(self ,
                 dim , 
                 num_gates , 
                 eps = 1e-9 ,
                 outer_expert_dim = tuple() ,
                 second_policy_train = 'random', 
                 second_policy_eval = 'random', 
                 second_threshold_train = 0.2,
                 second_threshold_eval = 0.2,
                 capacity_factor_train = 1.25,
                 capacity_factor_eval = 1.25):
        super(TopKGate , self).__init__()
        self.eps = eps
        self.num_gayes = num_gates 
        self.w_gating = nn.Parameter(torch.randn(outer_expert_dim , dim , num_gates))
        
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        
    def forward(self , x , important = None):
        pass
        
       
        
