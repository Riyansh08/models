# NanoGPT MODEL WITH Flash Attention 
import torch
import torch.nn as nn  
import math  
import inspect 
import numpy as np
from torch.nn import functional as F 
from dataclasses import dataclass 
import wandb # Weight and Bias 
from torch.nn import Embedding
from typing import Optional 
@dataclass
class NanoGPTConfig:
    batch_size : int = 8
    n_embd:int = 1600
    norm_eps: float = 1e-5
    norm_bias : bool = True
    attention_bias : bool = False 
    batch_norm_momentum : float = 0.999
    n_layer : int  = 48
    n_head : int = 25
    vocab_size : int = 50257
    block_size : int = 1024 # Context Length
    dropout : float = 0.01
    device : str = 'cuda' if torch.cuda.is_available else 'cpu'
    
# 1558M parameters

#LayerNormalization 

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias"""
    def __init__(self , config : NanoGPTConfig):
        super(LayerNorm , self).__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))
        self.eps = config.norm_eps 
    def forward(self , x):
        mean = x.mean(dim= -1 , keepdim = True)
        variance = x.var(dim = -1 , keepdim = True)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return ( self.weight * x_norm ) + self.bias if self.config.norm_bias else self.weight * x_norm 
#Used Earlier using Running Mean and variance
class BatchNorm(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(BatchNorm , self).__init__()
        self.config = config 
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))
        self.eps = config.norm_eps 
        self.register_buffer('running_mean' , torch.zeros(config.n_embd))
        self.register_buffer('running_variance' , torch.ones(config.n_embd))
        self.momentum = config.batch_norm_momentum
    def forward(self , x):
        if self.training:
          batch_mean = x.mean(dim = 0 , keepdim = True)
          batch_variance = x.var(dim = 0 , keepdim = True)
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
          self.running_variance = (1 - self.momentum) * self. running_variance + self.momentum * batch_variance
          x_norm = (x - batch_mean) / torch.sqrt(batch_variance + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_variance + self.eps)
        return (self.weight * x_norm + self.bias) if self.config.norm_bias else self.weight * x_norm

#Activation Function 
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU , self).__init__()
    def forward(self , x):
        return (0.5 * x ) * ( 1.0 + torch.erf(x / math.sqrt(2.0)))

class SwiGeLU(nn.Module):
    def __init__(self):
        super(SwiGeLU , self).__init__()
    def forward(self , x ):
        return x * torch.sigmoid(1.702 * x)
#Multi - Layer - Perceptron 

class MLP(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(MLP , self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.n_embd , 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd  , config.n_embd)
        self.act = SwiGeLU()
    def forward(self , x):
        return self.fc2(self.act(self.fc1(x)))
#Casual Self-Attention 
# Flash Attention 
class CasualSelfAttention(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(CasualSelfAttention , self).__init__()
        self.config = config 
        assert config.n_embd % config.n_head == 0 , "dimensions must be divisible by number of heads"
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.attention_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout 
        # QKV Projections 
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd  , bias = config.attention_bias)
        self.c_proj = nn.Linear(config.n_embd , config.n_embd , bias = config.attention_bias)
        # Check if Flash Attention is available
        self.flash = hasattr(torch.nn.functional , 'scaled_dot_product_attention')
        if not self.flash:
            print('WARNING : Flash attention is not available , using slow attention ')
            self.register_buffer('casual_mask' , torch.tril(torch.ones(config.block_size , config.block_size)).view(1 , 1 , config.block_size , config.block_size))
        #Forward 
        def forward(self , x ): 
            B , T , C = x.size()
            q , k , v = self.c_attn(x).split(self.n_embd , dim = -1)
            k = k.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 )
            v = v.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 ) # B , n_head  , T , C // n_head
            q = q.view(B , T , self.n_head , C // self.n_head).transpose(1 , 2 ) #  B , n_head  , T , C // n_head
            
            if self.flash:
                out = torch.nn.functional.scaled_dot_product_attention(q , k , v , attn_mask = None , dropout = self.dropout if self.training else 0.0 , is_casual = True)
            else:
                #Slow Attention 
                # k becomes B , n_head , C // n_head, T
                att = (q @ k.transpose(-2 , -1 )) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.casual_mask[: , : , :T , :T ] == 0 , float('-inf'))
                att = F.softmax(att , dim = -1)
                att = self.attention_dropout(att)
                # B , n_head , T , T @ B , n_head , T ,C //n_head -> B  , n_head , T , C// n_head
                out = att @ v
            
            out = out.transpose(1 , 2 ).contiguous().view(B , T , C)
            out = self.c_proj(out)
            return out
            print("Done")
        
       
class Block(nn.Module):
    def __init__(self , config):      
            super(Block , self).__init__()
            self.ln_1 = LayerNorm(config)
            self.attention = CasualSelfAttention(config)
            self.ln_2 = LayerNorm(config)
            self.mlp = MLP(config)
    def forward(self , x ):
        # Pre-Normalization
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class NanoGPT(nn.Module):
    def __init__(self , config : NanoGPTConfig):
        super(NanoGPT , self).__init__()
        self.config = config
        assert config.vocab_size is not None , "vocab_size is required"
        assert config.block_size is not None , "Context Length is required"
        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(config.vocab_size , config.n_embd), 
            pos_embeding = nn.Parameter(torch.zeros(1 ,config.block_size , config.n_embd)), 
            drop = nn.Dropout(config.dropout), 
            blocks = nn.ModuleList([Block(config) for _ in range (config.n_layer)]), 
            final_layer_norm = LayerNorm(config)
        ))
        self.output_logits = nn.Linear(config.n_embd , config.vocab_size)
        #initialize output projection weights
        self.apply(self._init_weights)
        for pn , p in self.named_parameters(): # Pytorch NN function
            if pn.endswith('c_proj_weight'):
                torch.nn.init.normal(p , mean = 0.0 , std = 0.02/math.sqrt(2 * config.n_layer))
        print("Model initalized")
        print("Number of parameters : %.2fM " % (self.get_num_params() / 1e6))
    
    def get_num_params(self , non_embeddings = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            n_params-= self.transformer.embedding.weight.numel()
        return n_params 
    # Initialize all weights 
    def _init_weights(self , module):
        if isinstance(module , nn.Linear):
            torch.nn.init.normal_(module.weight , mean = 0.0 , std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module , nn.Embedding):
            torch.nn.init.normal_(module.weight , mean = 0.0 , std = 0.02)
    
    # Forward method 
    def forward(self , idx , targets = None): 
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size , "T should be less than context length " 
        pos = torch.arrange(0 , T , dtype = torch.long , device = device) 
        # Start the forward pass 
        token_embeddings = self.transformer.embedding(idx)
        pos_embeddings = self.transformer.pos_embedding(pos)
        x = token_embeddings + pos_embeddings 
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.final_layer_norm(x)
        if targets is not None: 
            logits = self.output_logits(x)
            loss = F.cross_entropy(logits.view(-1 , logits.size(-1)) , targets.view(-1) , ignore_index = -1)
        else: 
            logits = self.output_logits(x[: , [-1] ,  : ]) #Note : [-1] is used to get the last token 
            loss = None
        return logits , loss 
    
    def crop_block_size(self , block_size):
        assert block_size <= self.config.block_size , "Cropping block size should be less than the context length"
        self.block_size = block_size 
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size]) 
        for block in self.transformer.blocks:
            if hasattr(block.attention , 'casual_mask'):
                block.attention.casual_mask = block.attention.casual_mask[: , : , :block_size , :block_size]
    
    @classmethod 
    # Method belongs the the class not the instance object 
    def from_pretrained(cls ,model_type ,  config: Optional[NanoGPTConfig] = None , override_args = None):
        assert model_type in {'gpt2' , 'gpt2-medium' , 'gpt2-large' , 'gpt2-xl'}
        override_args = override_args or {}
        assert all( k == "dropout" for k in override_args) , "only dropout can be overridden"
        from transformers import GPT2LMHeadModel , GPT2Config
        print(f"Loading weights from transformers library")
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
             'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
             'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab size : 50257 , block size : 1024 , bias : False")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['attention_bias'] = False 
        if 'dropout ' in override_args:
            config_args['dropout'] = override_args['dropout']
        config = NanoGPTConfig(**config_args)
        model = NanoGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_HF = GPT2LMHeadModel.from_pretrained(model_type)
        sd_HF  = model_HF.state_dict()
        sd_keys_HF = sd_HF.keys()
        sd_keys_HF = [k for k in sd_keys_HF if not k.endswith('.attn,bias' &'.attn.masked_bias')]
        transposed = ['attn.c_attn.weight' , 'attn.c_proj.weight' , 'mlp.c_fc.weight' , 'mlp.c_proj.weight']
        assert len(sd_keys_HF) == len(sd_keys) , f"Mismatch in number of keys : {len(sd_keys_HF)} != {len(sd_keys)}"
        
        
        
    
    def estimate_flops(self):
        pass
    def configure_optimizers(self):
        pass 
    
    @torch.no_grad()
    def generate(self):
        pass