# dpo.py
# Data : https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized
# WITH FIXED ERRORS FROM FINE-TUNE DPO 
#CAN RUN THIS CODE IN JUPYTER NOTEBOOK 

import json
import random 
import torch   
import torch.nn as nn 
import torch.nn.functional as F
import math  
from dataclasses import dataclass 
from tokenizers import Tokenizer 
from pathlib import Path 
from torch.utils.data import Dataset, DataLoader 
import wandb 
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class Config:
    dpo_model_name: str = "mistralai/Mistral-7B-Instruct"
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer_name: str = "mistralai/Mistral-7B-Instruct"
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"
    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lr: float = 2e-6
    weight_decay: float = 0.0
    epochs: int = 1
    max_length: int = 512
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class DPOConfig:
    beta: float = 0.1

class PairwiseDataset(Dataset):
    def __init__(self, file_path, tokenizer, config: Config):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        chosen = example['chosen']
        rejected = example['rejected']
        inputs = self.tokenizer(
            chosen, truncation=True, max_length=self.config.max_length, padding='max_length', return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                chosen, truncation=True, max_length=self.config.max_length, padding='max_length', return_tensors="pt"
            )
        inputs_r = self.tokenizer(
            rejected, truncation=True, max_length=self.config.max_length, padding='max_length', return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels_r = self.tokenizer(
                rejected, truncation=True, max_length=self.config.max_length, padding='max_length', return_tensors="pt"
            )
        batch = {
            "input_ids": inputs.input_ids.squeeze(), 
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze(),
            "input_ids_r": inputs_r.input_ids.squeeze(), 
            "attention_mask_r": inputs_r.attention_mask.squeeze(),
            "labels_r": labels_r.input_ids.squeeze(),
        }
        return batch

class DPO:
    def __init__(self, ref_model, policy_model, config: DPOConfig, device: str):
        self.ref_model = ref_model.to(device)
        self.policy_model = policy_model.to(device)
        self.beta = config.beta
        self.device = device

    def DPOloss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        input_ids_r = batch["input_ids_r"].to(self.device)
        attention_mask_r = batch["attention_mask_r"].to(self.device)

        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            ref_logits = ref_outputs.logits
            ref_outputs_r = self.ref_model(
                input_ids=input_ids_r, attention_mask=attention_mask_r, labels=input_ids_r
            )
            ref_logits_r = ref_outputs_r.logits

        policy_outputs = self.policy_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        policy_logits = policy_outputs.logits
        policy_outputs_r = self.policy_model(
            input_ids=input_ids_r, attention_mask=attention_mask_r, labels=input_ids_r
        )
        policy_logits_r = policy_outputs_r.logits

        # compute log-prob
        ll_ref = -F.cross_entropy(
            ref_logits.view(-1, ref_logits.size(-1)),
            input_ids.view(-1),
            reduction='none'
        ).view(input_ids.size())
        ll_ref_r = -F.cross_entropy(
            ref_logits_r.view(-1, ref_logits_r.size(-1)),
            input_ids_r.view(-1),
            reduction='none'
        ).view(input_ids_r.size())
        ll_policy = -F.cross_entropy(
            policy_logits.view(-1, policy_logits.size(-1)),
            input_ids.view(-1),
            reduction='none'
        ).view(input_ids.size())
        ll_policy_r = -F.cross_entropy(
            policy_logits_r.view(-1, policy_logits_r.size(-1)),
            input_ids_r.view(-1),
            reduction='none'
        ).view(input_ids_r.size())

        # sum across sequence
        sum_ll_ref = ll_ref.sum(-1)
        sum_ll_ref_r = ll_ref_r.sum(-1)
        sum_ll_policy = ll_policy.sum(-1)
        sum_ll_policy_r = ll_policy_r.sum(-1)

        # DPO loss
        logits = sum_ll_policy - sum_ll_policy_r
        ref_logits_diff = sum_ll_ref - sum_ll_ref_r
        clip_value = torch.clamp(logits - ref_logits_diff, min=-50, max=50)

        loss = -F.logsigmoid(self.beta * clip_value).mean()
        return loss

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    input_ids_r = torch.stack([b["input_ids_r"] for b in batch])
    attention_mask_r = torch.stack([b["attention_mask_r"] for b in batch])
    labels_r = torch.stack([b["labels_r"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_ids_r": input_ids_r,
        "attention_mask_r": attention_mask_r,
        "labels_r": labels_r,
    }

def main():
    config = Config()
    dpo_config = DPOConfig()

    # Initialize tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    ref_model = AutoModelForCausalLM.from_pretrained(config.dpo_model_name)
    policy_model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Load datasets
    train_dataset = PairwiseDataset(config.train_file, tokenizer, config)
    val_dataset = PairwiseDataset(config.val_file, tokenizer, config)
    train_loader = DataLoader(train_dataset, batch_size=config.micro_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.micro_batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # DPO loss wrapper
    dpo = DPO(ref_model, policy_model, dpo_config, config.device)

    # W&B initialization
    wandb.init(project="dpo-training", config=config.__dict__)

    global_step = 0
    for epoch in range(config.epochs):
        policy_model.train()
        for step, batch in enumerate(train_loader):
            loss = dpo.DPOloss(batch)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to W&B
                wandb.log({"train/loss": loss.item(), "step": global_step})

        # Validation
        policy_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_loss = dpo.DPOloss(batch)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        wandb.log({"val/loss": avg_val_loss, "epoch": epoch})

    # Save the fine-tuned policy model
    policy_model.save_pretrained("dpo_finetuned_model")
    tokenizer.save_pretrained("dpo_finetuned_model")

if __name__ == "__main__":
    main()
