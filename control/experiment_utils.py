import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
from control.env_config import BATCH_SIZE

def load_model_and_tags(model_name_or_path):
    """
    Loads the model and tokenizer, and returns user/assistant tags for prompt formatting.
    """
    if "qwen" in model_name_or_path.lower():
        user_tag = "<|im_start|>user\n"
        assistant_tag = "<|im_end|>\n<|im_start|>assistant/no_think\n<think>\n</think>\n"
        assistant_prompt_for_choice = "Answer: "
    elif "mistral" in model_name_or_path.lower():
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
        assistant_prompt_for_choice = "Answer: "
    else:
        user_tag = "USER: "
        assistant_tag = "ASSISTANT: "
        assistant_prompt_for_choice = "Answer: "
    try:
        if 'gpt' in model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left", legacy=False)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return None, None, None, None, None, None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = model.device
    return model, tokenizer, user_tag, assistant_tag, assistant_prompt_for_choice, device

def plot_pca_variance(pca, plot_dir, title="Percentage of Variance by PC", filename="pca_variance.png"):
    """
    Plots a histogram/bar plot of percentage of variance explained by each principal component.
    """
    explained_var = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_var)+1), explained_var, color='skyblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage of Variance Explained')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
    print(f"Saved PCA variance plot to {os.path.join(plot_dir, filename)}")

def evaluate_repread_accuracy(rep_reading_pipeline, dataset, rep_token, hidden_layers, rep_reader, plot_dir, debug=False):
    """
    Evaluates RepReader accuracy and plots accuracy by layer.

    Two strategies:
    - If dataset['test']['labels'] is available and length-matches the test data, compute accuracy
      by comparing sign(direction · activation) to the label for each sample.
    - Otherwise, fall back to the legacy pairwise ordering assumption (adjacent pairs belong together).
    """
    test_data = dataset.get('test', {}).get('data', [])
    test_labels = dataset.get('test', {}).get('labels')
    H_tests = rep_reading_pipeline(test_data, rep_token=rep_token, hidden_layers=hidden_layers, rep_reader=rep_reader, batch_size=BATCH_SIZE)
    results = {}

    use_label_mode = isinstance(test_labels, (list, tuple)) and len(test_labels) == len(H_tests)
    if debug:
        uniq = set(test_labels) if use_label_mode else set()
        print(f"[DEBUG] RepReader eval using {'label' if use_label_mode else 'pairwise'} mode; labels unique={sorted(list(uniq)) if uniq else 'n/a'}")

    for layer in hidden_layers:
        try:
            sign = float(rep_reader.direction_signs[layer][0])
        except Exception:
            sign = 1.0

        if use_label_mode:
            # Map labels to binary {0,1}
            accs = []
            for H, y in zip(H_tests, test_labels):
                v = H[layer]
                # Normalize potential list/ndarray to scalar
                if isinstance(v, (list, tuple, np.ndarray)):
                    v = float(np.array(v).reshape(-1)[0])
                try:
                    y_bin = int(y)
                    if y_bin not in (0, 1):
                        # Treat negative as 0, positive as 1
                        y_bin = 1 if float(y) > 0 else 0
                except Exception:
                    y_bin = 0
                pred_bin = 1 if (float(v) * sign) > 0 else 0
                accs.append(1.0 if pred_bin == y_bin else 0.0)
            results[layer] = float(np.mean(accs)) if accs else 0.0
        else:
            # Legacy: assume adjacent pairs belong together
            H_test_layer = [H[layer] for H in H_tests]
            if len(H_test_layer) % 2 != 0:
                H_test_layer = H_test_layer[:-1]
            if not H_test_layer:
                results[layer] = 0.0
                continue
            # Normalize potential arrays to scalars
            H_scalars = []
            for v in H_test_layer:
                if isinstance(v, (list, tuple, np.ndarray)):
                    v = float(np.array(v).reshape(-1)[0])
                H_scalars.append(float(v))
            H_test_pairs = np.array([H_scalars[i:i+2] for i in range(0, len(H_scalars), 2)])
            if not H_test_pairs.size:
                results[layer] = 0.0
                continue
            cors = np.mean(H_test_pairs[:, 0] > H_test_pairs[:, 1]) if sign >= 0 else np.mean(H_test_pairs[:, 0] < H_test_pairs[:, 1])
            results[layer] = float(cors)

    best_layer = max(results, key=results.get) if results else None
    if debug and best_layer is not None:
        print(f"[DEBUG] Best layer: {best_layer} with accuracy {results.get(best_layer, 0):.4f}")
    # Plot
    try:
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(hidden_layers, [results.get(l, 0.0) for l in hidden_layers], marker='o', linestyle='-', color='blue')
        plt.title("RepReader Accuracy by Layer")
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.xticks(hidden_layers, rotation=45)
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(plot_dir, "rep_reader_accuracy_by_layer.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved RepReader accuracy plot to {out_path}")
    except Exception as e:
        print(f"Warning: failed to save RepReader accuracy plot: {e}")
    return results, best_layer
 
