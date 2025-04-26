#Importing necassary libraries
#AUTHOR - Riyansh Shah
#CODE FOR INFERENCE AND SAMPLING 
import torch
from transformers import AutoTokenizer
from model import KVCache , LanguageModel
from utils import load_model
import argparse 
#MOVE DEVICE 
def move_to_device(inputs: dict, device: str = torch.cuda if torch.cuda.is_available() else torch.cpu) -> dict:
    return {k: v.to(device) for k, v in inputs.items()}
#CUSTOM SAMPLING METHOD 
def min_p_sampling(logits , p_base = 0.9): #CHOSING THE BEST NEXT TOKEN 
    p_max = logits.max(dim = -1 , keepdim = True).Values()
    p_scaled = p_max * p_base
    mask = logits >= p_scaled
    logits = logits * mask.float()
    logits = logits / logits.sum(dim = -1 , keepdim = True)
    next_token = torch.multinomial(logits, num_samples = 1).item()
    return next_token
# #Sampling function - native , Normal 
# def sample(model: LanguageModel, tokenizer: AutoTokenizer, prompt: str, max_length: int, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, num_return_sequences: int = 1):
#     inputs = tokenizer(prompt, return_tensors="pt").input_ids
#     inputs = move_to_device(inputs)
#     outputs = model.generate(inputs, max_length=max_length, temperature=temperature, top_p=top_p, top_k=top_k, num_return_sequences=num_return_sequences)
#     return tokenizer.batch_decode(outputs, skip_special_tokens=True)

#INFERENCE CODE
def inference(model: LanguageModel, tokenizer: AutoTokenizer, prompt: str, max_length: int, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, num_return_sequences: int = 1):
    inputs = tokenizer(prompt , return_tensors = 'pt')
    inputs['position_ids'] = inputs['attention_mask'].cumsum(dim = -1)
    inputs = move_to_device(inputs)
    kv_cache = KVCache()
    stop_token = tokenizer.eos_token_id
    generated_tokens = []
    # Importing necessary libraries
# AUTHOR - Riyansh Shah
# CODE FOR INFERENCE AND SAMPLING

import torch
from transformers import AutoTokenizer
from model import KVCache, LanguageModel
from utils import load_model
import argparse

# MOVE TO DEVICE
def move_to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in inputs.items()}

# CUSTOM SAMPLING METHOD
def min_p_sampling(probs: torch.Tensor, p_base: float = 0.9) -> int:
    # Keep only tokens whose probability ≥ p_base * max_prob
    p_max = probs.max(dim=-1, keepdim=True).values
    threshold = p_max * p_base
    mask = probs >= threshold
    pruned = probs * mask.float()
    pruned = pruned / pruned.sum(dim=-1, keepdim=True)
    # Sample one token ID
    token = torch.multinomial(pruned, num_samples=1)
    return token.item()

# INFERENCE CODE
def inference(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int,
    temperature: float = 0.7,
    do_sample: bool = True,
    p_base: float = 0.9,
    device: torch.device = torch.device("cpu")
) -> str:
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Build position IDs from attention mask
    inputs["position_ids"] = inputs["attention_mask"].cumsum(dim=-1)
    inputs = move_to_device(inputs, device)

    kv_cache = KVCache()                     # empty cache
    stop_token = tokenizer.eos_token_id      # end-of-sequence ID
    generated_tokens = []

    for _ in range(max_length):
        # Forward pass with cache
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            kv_cache=kv_cache
        )
        kv_cache = outputs["kv_cache"]       # update cache
        logits = outputs["logits"][:, -1, :] # scores for last position

        if do_sample:
            # apply temperature and sample
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = min_p_sampling(probs, p_base=p_base)
        else:
            # greedy: pick highest logit
            next_id = int(logits.argmax(dim=-1).item())

        generated_tokens.append(next_id)
        if next_id == stop_token:
            break

        # prepare next inputs
        inputs["input_ids"] = torch.tensor([[next_id]], device=device)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1,1), device=device)], dim=-1
        )
        inputs["position_ids"] = inputs["attention_mask"].cumsum(dim=-1)

    # decode and return
    return prompt + tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Model Inference")
    parser.add_argument("--prompt",       type=str,   required=True, help="The input prompt")
    parser.add_argument("--max_length",   type=int,   default=100,    help="Max tokens to generate")
    parser.add_argument("--temperature",  type=float, default=0.7,    help="Sampling temperature")
    parser.add_argument("--do_sample",    action="store_true",       help="Enable sampling (otherwise greedy)")
    parser.add_argument("--p_base",       type=float, default=0.9,    help="Relative probability cutoff for sampling")
    parser.add_argument("--model_path",   type=str,   default="../gemma-2b", help="Path to Gemma model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, device)
    model.to(device).eval()
    print("Model loaded. Generating...")

    with torch.no_grad():
        result = inference(
            model,
            tokenizer,
            args.prompt,
            args.max_length,
            temperature=args.temperature,
            do_sample=args.do_sample,
            p_base=args.p_base,
            device=device
        )
    print(result)

 