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
    
    def __init__(self , dimension , num_heads , dropout : 0.0 , bias = False):
        super(MultiHeadAttention , self ).__init__()
        
        self.dimension= dimension 
        self.num_heads = num_heads
        assert dimension % num_heads == 0, "dimension must be divisible by num_heads"
        
        self.head_dim = dimension // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Initialize projection layers
        
        self.w_q = nn.Linear(dimension, dimension, bias=bias)
        self.w_k = nn.Linear(dimension, dimension, bias=bias)
        self.w_v = nn.Linear(dimension, dimension, bias=bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.attention = ScaledDotProduct()
        
    # Forward pass
    def forward(self , x , mask : None , output_attentions : False):
        q = v = k = x
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.q_v(v)
         
        # Reshape the input for multi-head attention
        key = key.view(key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1 , 2)
        query = query.view(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1 , 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x , attention_weights = self.attention(key , query , value , mask , self.dropout)
        # Reshape the output back to the original shape
        x = x.transpose(1 , 2).contigious().view(x.size(0), -1, self.head_dim * self.num_heads)
        
        x = self.w_o(x)
        x = self.dropout(x)
        if output_attentions:
            return x , attention_weights
        else:
         return x 
     
#positional encoding - CLIP uses sinusoidal positional encoding 