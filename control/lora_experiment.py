import os
import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# Minimal dataset for LoRA fine-tuning using positive authority bias examples only
class PositiveAuthorityDataset(Dataset):
    """(Deprecated name) Dataset wrapping positive examples for causal LM fine-tuning.

    This class name is retained for backward compatibility. Use PositiveBiasDataset instead.
    """
    def __init__(self, examples: List[str], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        # Casual LM: labels are input_ids shifted internally by model
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

def make_collate_fn(pad_token_id: int):
    def _collate(batch):
        # Pad to max length in batch
        max_len = max(item['input_ids'].size(0) for item in batch)
        input_ids_list, labels_list, mask_list = [], [], []
        for item in batch:
            pad_len = max_len - item['input_ids'].size(0)
            if pad_len > 0:
                input_ids_list.append(torch.cat([item['input_ids'], torch.full((pad_len,), fill_value=pad_token_id, dtype=torch.long)]) )
                labels_pad = torch.full((pad_len,), fill_value=-100, dtype=torch.long)
                labels_list.append(torch.cat([item['labels'], labels_pad]))
                mask_list.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
            else:
                input_ids_list.append(item['input_ids'])
                labels_list.append(item['labels'])
                mask_list.append(item['attention_mask'])
        return {
            'input_ids': torch.stack(input_ids_list),
            'labels': torch.stack(labels_list),
            'attention_mask': torch.stack(mask_list)
        }
    return _collate

@dataclass
class LoRAFineTuneConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lr: float = 2e-4
    batch_size: int = 4
    max_steps: int = 200
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    output_dir: str = "lora_outputs"
    save_steps: int = 100
    max_length: int = 768
    dtype: str = "bfloat16"
    record_task_vector: bool = True  # store weight deltas (concept vector)
    task_vector_filename: str = "task_vector.pt"

class LoRAFineTuner:
    def __init__(self, base_model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: LoRAFineTuneConfig):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Will hold a snapshot of base weights for target modules for revert logic
        self._initial_base_snapshot: Dict[str, torch.Tensor] = {}

    def _prepare_model(self):
        # Cast to appropriate dtype
        target_dtype = torch.bfloat16 if self.cfg.dtype == 'bfloat16' and torch.cuda.is_available() else torch.float16
        self.base_model.to(target_dtype)
        lora_config = LoraConfig(
            r=self.cfg.r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias=self.cfg.bias,
            task_type=self.cfg.task_type,
            target_modules=list(self.cfg.target_modules)
        )
        peft_model = get_peft_model(self.base_model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def train(self, train_dataset: Dataset, return_deltas: bool = False) -> str | Tuple[str, Optional[Dict[str, torch.Tensor]]]:
        # FIRST: Capture base weights BEFORE preparing LoRA model
        base_snapshot = {}
        if self.cfg.record_task_vector:
            target_substrings = set(self.cfg.target_modules)
            for name, param in self.base_model.named_parameters():
                if any(t in name for t in target_substrings):
                    base_snapshot[name] = param.detach().clone().to('cpu')
            print(f"[LoRA] Captured {len(base_snapshot)} base model parameters for task vector")
            print(f"[LoRA] Sample base parameter names: {list(base_snapshot.keys())[:5]}")
            # Keep an internal copy for potential revert
            self._initial_base_snapshot = {k: v.clone() for k, v in base_snapshot.items()}
        
        # THEN: Prepare LoRA model
        model = self._prepare_model()
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=make_collate_fn(self.tokenizer.pad_token_id))

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        total_steps = self.cfg.max_steps
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        global_step = 0
        losses = []
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        while global_step < self.cfg.max_steps:
            for batch in train_loader:
                if global_step >= self.cfg.max_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / self.cfg.gradient_accumulation_steps
                loss.backward()
                losses.append(loss.item())

                if (global_step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if global_step % self.cfg.logging_steps == 0:
                    recent_losses = losses[-self.cfg.logging_steps:]
                    avg_loss = sum(recent_losses) / max(1, len(recent_losses))
                    print(f"[LoRA] step={global_step} loss={avg_loss:.4f}")

                global_step += 1
                if global_step % self.cfg.save_steps == 0 or global_step >= self.cfg.max_steps:
                    save_dir = 'final' if global_step >= self.cfg.max_steps else f"checkpoint-{global_step}"
                    save_path = os.path.join(self.cfg.output_dir, save_dir)
                    model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
                        json.dump({'step': global_step, 'loss': losses[-1]}, f)
                    if global_step >= self.cfg.max_steps:
                        print(f"[LoRA] Training complete. Saved to {save_path}")
                        # After finishing training, optionally compute task vector
                        deltas = None
                        if self.cfg.record_task_vector:
                            deltas = self._compute_and_save_task_vector(model, base_snapshot, save_path)
                        # If caller wants deltas but to keep base model unchanged, revert base weights
                        if return_deltas and self._initial_base_snapshot:
                            self._revert_base_weights()
                        return (save_path, deltas) if return_deltas else save_path
        final_dir = os.path.join(self.cfg.output_dir, 'final')
        if self.cfg.record_task_vector:
            deltas = self._compute_and_save_task_vector(model, base_snapshot, final_dir)
        else:
            deltas = None
        if return_deltas and self._initial_base_snapshot:
            self._revert_base_weights()
        return (final_dir, deltas) if return_deltas else final_dir

    def _compute_and_save_task_vector(self, peft_model, base_snapshot, save_path) -> Dict[str, torch.Tensor]:
        try:
            merged = peft_model.merge_and_unload()
        except Exception:
            # If already merged
            merged = peft_model
        deltas = {}
        
        # Debug: Print parameter names to understand the mismatch
        print(f"[LoRA] Base snapshot has {len(base_snapshot)} parameters")
        print(f"[LoRA] Merged model has {len(list(merged.named_parameters()))} parameters")
        
        base_param_names = set(base_snapshot.keys())
        merged_param_names = set(name for name, _ in merged.named_parameters())
        
        if len(base_param_names.intersection(merged_param_names)) == 0:
            print(f"[LoRA] WARNING: No parameter name overlap found!")
            print(f"[LoRA] Base sample names: {list(base_param_names)[:5]}")
            print(f"[LoRA] Merged sample names: {list(merged_param_names)[:5]}")
        
        for name, param in merged.named_parameters():
            if name in base_snapshot:
                base_w = base_snapshot[name].to(param.dtype).to(param.device)
                delta = param.detach() - base_w
                deltas[name] = delta.cpu()
            else:
                # Check if there's a close match (LoRA might have changed parameter names)
                target_substrings = set(self.cfg.target_modules)
                if any(t in name for t in target_substrings):
                    print(f"[LoRA] WARNING: Parameter {name} not found in base snapshot but matches target modules")
        
        task_vector_path = os.path.join(save_path, self.cfg.task_vector_filename)
        torch.save({
                'deltas': deltas,
                'base_param_names': list(deltas.keys()),
                'config': {
                    'target_modules': self.cfg.target_modules,
                    'r': self.cfg.r,
                    'alpha': self.cfg.lora_alpha
                }
            }, task_vector_path)
        print(f"[LoRA] Task vector saved to {task_vector_path} with {len(deltas)} tensors")
        return deltas

    def _revert_base_weights(self):
        """Revert base model target module weights back to the initially captured snapshot."""
        if not self._initial_base_snapshot:
            return
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self._initial_base_snapshot:
                    param.copy_(self._initial_base_snapshot[name].to(param.device, param.dtype))
        print("[LoRA] Reverted base model target module weights to original snapshot")

# Utility to extract ONLY positive authority biased examples from existing dataset creation logic

def extract_positive_examples(dataset: Dict[str, Any]) -> List[str]:
    """Generic positive example extractor for ANY bias dataset with pair labels.

    Expectations:
    - dataset['train']['data'] is a flat list where every successive pair of entries belongs together.
    - dataset['train']['labels'] is a list of length N_pairs; each element is a list/tuple of two booleans
      indicating which member of the pair is the preferred / positive example (e.g. [True, False]).

    Returns a list of the textual positive examples that pass minimal completion filtering.
    """
    positives = []
    filtered_pairs = 0
    def _is_complete(text: str) -> bool:
        markers = [
            "I completely agree",
            "I mostly agree",
            "I'm neutral",
            "I mostly disagree",
            "I completely disagree",
            "I choose option",
        ]
        lower = text.lower()
        return any(m.lower() in lower for m in markers)
    train_data = dataset.get('train', {}).get('data', [])
    train_labels = dataset.get('train', {}).get('labels', [])
    if not train_data or not train_labels:
        return positives
    assert len(train_labels) * 2 <= len(train_data), "Mismatch between labels and data length for pair reconstruction"
    idx = 0
    for lbl_pair in train_labels:
        pair_items = train_data[idx:idx+2]
        idx += 2
        if len(pair_items) < 2:
            continue
        chosen = None
        if lbl_pair[0]:
            chosen = pair_items[0]
        elif lbl_pair[1]:
            chosen = pair_items[1]
        if chosen and _is_complete(chosen):
            positives.append(chosen)
        else:
            filtered_pairs += 1
    if filtered_pairs:
        print(f"[LoRA] Filtered {filtered_pairs} positive label pairs due to incomplete answers")
    return positives

def extract_negative_authority_examples(dataset: Dict[str, Any]) -> List[str]:
    """Extract the negative member of each labeled pair (opposite of the positive one)."""
    negatives = []
    filtered_pairs = 0
    def _is_complete(text: str) -> bool:
        markers = [
            "I completely agree",
            "I mostly agree",
            "I'm neutral",
            "I mostly disagree",
            "I completely disagree",
            "I choose option",
        ]
        lower = text.lower()
        return any(m.lower() in lower for m in markers)
    train_data = dataset.get('train', {}).get('data', [])
    train_labels = dataset.get('train', {}).get('labels', [])
    if not train_data or not train_labels:
        return negatives
    assert len(train_labels) * 2 <= len(train_data), "Mismatch between labels and data length for pair reconstruction"
    idx = 0
    for lbl_pair in train_labels:
        pair_items = train_data[idx:idx+2]
        idx += 2
        if len(pair_items) < 2:
            continue
        # lbl_pair entries are booleans; True marks the positive element
        chosen = None
        if lbl_pair[0]:
            chosen = pair_items[1]
        elif lbl_pair[1]:
            chosen = pair_items[0]
        if chosen and _is_complete(chosen):
            negatives.append(chosen)
        else:
            filtered_pairs += 1
    if filtered_pairs:
        print(f"[LoRA] Filtered {filtered_pairs} negative label pairs due to incomplete answers")
    return negatives

#############################################
# Backwards compatibility aliases            #
#############################################

# Old function name some scripts may still import
def extract_positive_authority_examples(dataset: Dict[str, Any]) -> List[str]:  # pragma: no cover - thin wrapper
    return extract_positive_examples(dataset)

# Provide a more generic dataset class alias
class PositiveBiasDataset(PositiveAuthorityDataset):
    """Alias of PositiveAuthorityDataset (generic naming)."""
    pass

__all__ = [
    'LoRAFineTuneConfig', 'LoRAFineTuner',
    'PositiveAuthorityDataset', 'PositiveBiasDataset',
    'extract_positive_examples', 'extract_positive_authority_examples'
]
