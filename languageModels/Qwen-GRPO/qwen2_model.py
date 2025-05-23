import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math
import numpy as np
from inspect import isfunction
from dataclasses import dataclass , field , asdict
from pathlib import Path
from typing import Optional , Dict , Any , Tuple , List

@dataclass 
class Config:
    vocab_size :int =  151643
    use_cache : bool = True
    model_type : str = "Qwen2"
    bos_token_id : int =151643
    eos_token_id : int =  151645
    #MODEL ARCHITECTURE CONFIGS
    d_model : int = 4096 
    intermediate_size : int =11008
    n_layers : int = 32 
    n_head : int = 32 
    num_key_value_heads : int = 2 # if grouped_query_attention is True
    #REGULARIZATION CONFIGS 
    attention_dropout : float = 0.0 
    initializer_range : float = 0.02
    rms_norm_eps : float = 1e-5
    #EMBEDDING CONFIGS
    rope_theta: float = 1000000.0
    max_position_embeddings : int = 32768
    #SLIDING WINDOW CONFIGS 
    use_sliding_window : bool = False
    sliding_window_size : int = 32768
    max_windom_layers : int = 70 
    #RUNTIME CONFIGS 
    use_cache : bool = True
    tie_word_embeddings : bool = True
    torch_dtype : str = "bfloat16"