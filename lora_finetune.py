#!/usr/bin/env python3
"""Simple LoRA fine-tuning entrypoint focusing only on adapter training.

Usage example:
  python lora_finetune.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \
    --data-file data/authority/authority_MilgramS.json \
    --output-dir lora_outputs/mistral_authority \
    --num-epochs 1 --batch-size 1 --lr 2e-4

Notes:
- Only trains LoRA adapters; no RepE or prompt control is run here.
- Saves (a) LoRA adapter weights, (b) merged full model weights (optional),
  (c) training args & a tiny README metadata file.
"""
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------- Data Handling --------------
class SimpleTextDataset(Dataset):
    def __init__(self, samples: List[str], tokenizer, max_length: int = 1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        # Labels: same as input_ids for causal LM
        enc['labels'] = enc['input_ids'].clone()
        return enc

# -------------- Utility --------------

def load_authority_samples(json_path: str, limit: int = None) -> List[str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    for item in data:
        if isinstance(item, dict):
            # Try common fields
            scenario = item.get('scenario') or item.get('statement') or json.dumps(item)
        else:
            scenario = str(item)
        # Minimal instruction wrapping
        prompt = f"[INST] {scenario}\nAnswer: [/INST]"
        samples.append(prompt)
    if limit:
        samples = samples[:limit]
    return samples

# -------------- Training Core --------------

def train_lora(model, tokenizer, dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    model.train()
    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"Epoch {epoch+1} Step {global_step}/{total_steps} Loss {loss.item():.4f}")
    return model

# -------------- Main --------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning only")
    parser.add_argument('--base-model', type=str, required=True, help='Base model name or path')
    parser.add_argument('--data-file', type=str, required=True, help='Training JSON file (list of scenarios)')
    parser.add_argument('--output-dir', type=str, required=True, help='Where to save LoRA adapters and merged model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for quick tests')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--merge', action='store_true', help='Also save merged full model weights')
    parser.add_argument('--log-every', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer and base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None
    )

    # Prepare model (optional if quantization etc.)
    try:
        base_model = prepare_model_for_kbit_training(base_model)
    except Exception:
        pass

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM'
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    print("Loading training data...")
    samples = load_authority_samples(args.data_file, limit=args.limit)
    dataset = SimpleTextDataset(samples, tokenizer, max_length=args.max_length)
    print(f"Dataset size: {len(dataset)} samples")

    print("Starting LoRA training...")
    lora_model = train_lora(lora_model, tokenizer, dataset, args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    adapter_dir = os.path.join(args.output_dir, f"lora_adapters_{timestamp}")
    os.makedirs(adapter_dir, exist_ok=True)
    print(f"Saving LoRA adapters to {adapter_dir}")
    lora_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if args.merge:
        print("Merging LoRA weights into base model and saving full model...")
        merged_model = lora_model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, f"merged_model_{timestamp}")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")

    # Save run metadata
    meta = {
        'base_model': args.base_model,
        'data_file': args.data_file,
        'num_samples': len(dataset),
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'merged_saved': bool(args.merge),
        'timestamp': timestamp
    }
    with open(os.path.join(args.output_dir, f'run_meta_{timestamp}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(args.output_dir, 'README.txt'), 'a') as f:
        f.write(f"Run {timestamp}: {json.dumps(meta)}\n")

    print("Done. LoRA fine-tuning complete.")

if __name__ == '__main__':
    main()
