#LORA - Small implemtation 
import torch
import torch.nn as nn
class LoRA(nn.Module):
    def __init__(self , in_features , out_features , rank , alpha , en : bool = True):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(rank, out_features))
        self.B = nn.Parameter(torch.zeros(in_features , rank))
        self.scale = alpha/ rank 
        self.en = True
    def forward(self , weights):
        if self.en:
            return weights + torch.matmul(self.B , self.A).view(weights.shape) * self.scale
        else:
            return weights