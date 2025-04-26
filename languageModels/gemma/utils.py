#Importing necassary libraries
#AUTHOR - Riyansh Shah
#CODE FOR LOADING AND SAVING THE MODEL 
from transformers import AutoTokenizer
import glob 
from safetensors import safe_open 
import json 
import torch 
import torch.nn as nn
from typing import tuple
import os
from model import LanguageModel , Gemma , KVCache
from gemma.model import GemmaConfig

#LOADING THE MODEL 
def load_model(model_path:str | os.PathLike):
    tokenizer = AutoTokenizer.from_pretrained(model_path , padding_side="right")
    config = GemmaConfig.from_pretrained(model_path)
    model = Gemma(config)
    return tokenizer , model
#SAVING THE MODEL 
def save_model(model:nn.Module, model_path:str | os.PathLike):
    config = model.config
    config.save_pretrained(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))

    
    
    
