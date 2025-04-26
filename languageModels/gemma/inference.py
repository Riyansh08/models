#Importing necassary libraries
#AUTHOR - Riyansh Shah
#CODE FOR INFERENCE AND SAMPLING 
import torch
from transformers import AutoTokenizer
from model import KVCache , LanguageModel
from utils import load_model
import argparse 
#Sampling function 
def sample(model: LanguageModel, tokenizer: AutoTokenizer, prompt: str, max_length: int, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, num_return_sequences: int = 1):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length, temperature=temperature, top_p=top_p, top_k=top_k, num_return_sequences=num_return_sequences)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
