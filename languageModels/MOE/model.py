 #owner : Riyansh Shah 
 
# This is a simple implementation of a SPARSELY GATED Mixture of Experts (MoE) model in PyTorch.
# This is just THE MOE EXPERT implemetation. It does not have transformer CODE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from inspect import isFunction 


MIN_EXPERT_CAP  = 4



