import torch 
import torch.nn as nn
import math 
import torch.nn.functional as F
import numpy as np 
import pandas as pd

# to be implemented: CLIP model 

class SingleAttentionHead(nn.Module):
    
    def __init__(self , dimension , attention_head_size , dropout , bias = False):
        super(SingleAttentionHead, self).__init__()
        
        self.dimension = dimension 
        self.attention_head_size = attention_head_size
        self.dropout = dropout 
        
        self.query = nn.Linear(dimension , attention_head_size , bias = bias)
        self.key = nn.Linear(dimension, attention_head_size, bias=bias)
        self.value = nn.Linear(dimension, attention_head_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x ):
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
      query = self.query(x)
      key = self.key(x)
      value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
      attention_scores = torch.matmul(query, key.transpose(-1, -2))
      attention_scores_norm = attention_scores / math.sqrt(self.attention_head_size)
      attention_weights = F.softmax(attention_scores_norm , dim = -1)
      
      attention_weights = self.dropout(attention_weights)
      attention_output = torch.matmul(attention_weights , value)
      
      return ( attention_output , attention_weights)
      

        
     