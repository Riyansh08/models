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
  
class ScaledDotProduct(nn.Module):
    
    def __init__(self):
        super(ScaledDotProduct , self).__init__()
        
        self.softmax = nn.Softmax(dim = -1) # softmax 
        
    def forward(self , k , q , v , mask ,  dropout :nn.Dropout):
        
        d_k = q.size(-1) # Get the dimension of the query 
        scores = torch.matmul(q , k.transpose(-1 , -2) ) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = self.softmax(scores)
        if dropout:
         attention_weights = dropout(attention_weights)
        attention_output  =  torch.matmul(attention_weights , v)
        
        return ( attention_output , attention_weights ) 
    
    
class MultiHeadAttention(nn.Module):
    
    pass 
      

        
     