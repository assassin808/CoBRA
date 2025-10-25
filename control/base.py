# control/base.py

import torch
import torch.nn.functional as F # For softmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import sys
import json
from copy import deepcopy
import html
import matplotlib.colors as mcolors
import random # Added for permutation
import matplotlib.pyplot as plt
from tqdm import tqdm


class ControlExperiment:
    """A base class for running control experiments."""
    def __init__(self, model, tokenizer, device, plot_dir, is_testing_mode=False, num_permutations=5, experiment_suffix=""):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.plot_dir = plot_dir
        self.is_testing_mode = is_testing_mode
        self.num_permutations = num_permutations
        self.experiment_suffix = experiment_suffix  # For temperature ablation or other variations For temperature ablation or other variations

        if "qwen" in model.config.name_or_path.lower():
            self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant/no_think\n<think>\n</think>\n", "Answer: "
            self.thinking_assistant_tag = "<|im_end|>\n<|im_start|>assistant/think\n<think>\n"
        elif "mistral" in model.config.name_or_path.lower():
            self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice = "[INST]", "[/INST]", " Answer: "
            self.thinking_assistant_tag = "[/INST]"
        elif "llama" in model.config.name_or_path.lower():
            self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice = "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", " Answer: "
            self.thinking_assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif 'oss' in model.config.name_or_path.lower():
            self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice = "<|start|>user<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>", " Answer: "
            self.thinking_assistant_tag = "<|end|><|start|>assistant<|channel|>analysis<|message|>"
        elif 'seek' in model.config.name_or_path.lower():
            self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice = "<|begin▁of▁sentence|><|User|>\n", "<|Assistant|><think>\n</think>", "Answer: "
            self.thinking_assistant_tag = "<|Assistant|><think>\n"
        else:
            self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice = "USER: ", "ASSISTANT: ", "Answer: "
            self.thinking_assistant_tag = "ASSISTANT: " # Fallback

        self.mcq_scenarios = {
            "choice_first": "Choose one of the options (A, B, C, D, E) directly, e.g., 'Answer: X'.",
            "choice_then_reasoning": "Choose one of the options (A, B, C, D, E) and then briefly explain your reasoning, e.g., 'Answer: X. I think that...'.",
            "reasoning_then_choice": "First, very briefly explain your reasoning. Then, conclude by choosing one of the options (A, B, C, D, E) in the format 'Answer: X'.",
            "thinking_first": "First, very briefly explain your reasoning. Then, conclude by choosing one of the options (A, B, C, D, E) in the format 'Answer: X'."
        }

    def _get_target_token_map(self, choice_labels, option_to_number_map):
        target_choice_tokens_map = {}
        print(f"\n[INFO] Mapping choices to token variants (testing with {self.num_permutations} permutations per scenario)...")
        for label in choice_labels:
            number_label = option_to_number_map[label]
            token_variants = set()
            str_variants = [label, f" {label}", f"\t{label}", number_label, f" {number_label}"]
            for s in str_variants:
                try:
                    if s.strip():
                        encoded = self.tokenizer.encode(s, add_special_tokens=False)
                        if encoded: token_variants.add(encoded[0])
                except (IndexError, Exception) as e:
                    if self.is_testing_mode: print(f"[DEBUG] Could not encode variant '{s}' for label {label}: {e}")
                    continue
            target_choice_tokens_map[label] = list(token_variants)
            print(f"  - Choice '{label}': Mapped to token IDs {target_choice_tokens_map[label]}")
        return target_choice_tokens_map

    def _run_analysis(self, scenarios, mcq_options, prompt_template, control_levels, target_token_map, scenario_name, get_probs_fn, mcq_addon, **kwargs):
        all_probs = {level: {} for level in control_levels}
        detailed_results = []
        prompts_to_process = []
        for scenario in scenarios:
            for i in range(self.num_permutations):
                option_keys = list(mcq_options.keys())
                random.shuffle(option_keys)
                shuffled_options_dict = {k: mcq_options[k] for k in option_keys}
                options_block = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in shuffled_options_dict.items()])
                base_prompt = prompt_template.format(**scenario)
                for level in control_levels:
                    prompts_to_process.append({
                        "scenario": scenario, "permutation_id": i, "shuffled_options": shuffled_options_dict,
                        "options_block": options_block, "base_prompt": base_prompt, "level": level,
                    })
        get_probs_fn(prompts=prompts_to_process, batch_mode=True, mcq_prompt_addon=mcq_addon, target_token_map=target_token_map, **kwargs)
        for prompt_data in tqdm(prompts_to_process, desc=f"Processing {scenario_name} Scenarios ({kwargs.get('mcq_scenario_key', '')})"):
            level = prompt_data["level"]
            choice_probs, reasoning_text = get_probs_fn(prompts=[prompt_data], batch_mode=False, mcq_prompt_addon=mcq_addon, target_token_map=target_token_map, **kwargs)
            detailed_results.append({
                "scenario_id": prompt_data["scenario"].get("id"), "permutation_id": prompt_data["permutation_id"],
                "control_level": level, "operator": kwargs.get('operator'), "mcq_scenario": kwargs.get('mcq_scenario_key'),
                "base_prompt": prompt_data["base_prompt"], "shuffled_options": prompt_data["shuffled_options"],
                "reasoning": reasoning_text, "choice_probabilities": choice_probs
            })
            for label, prob in choice_probs.items():
                all_probs[level].setdefault(label, []).append(prob)
        avg_probs = {level: {label: np.mean(prob_list) if prob_list else 0.0 for label, prob_list in choice_data.items()} for level, choice_data in all_probs.items()}
        return avg_probs, detailed_results
    
    def _calculate_likert_from_probs(self, choice_probs, likert_weights):
        """Helper to calculate a single Likert score from a probability dictionary."""
        total_prob = sum(choice_probs.values())
        if total_prob == 0: return 2.0 # Return neutral if no valid probabilities
        rescaled_probs = {label: p / total_prob for label, p in choice_probs.items()}
        score = sum(rescaled_probs.get(label, 0) * likert_weights.get(label, 0) for label in likert_weights)
        return score

    def _plot_results(self, results, control_levels, choice_labels, scenario_name, control_type, plot_title, xlabel, xticks_labels=None, legend_options=None, mcq_scenario_key=""):
        fig, ax = plt.subplots(figsize=(14, 8))
        control_levels = np.array(control_levels)
        min_coeff, max_coeff = control_levels.min(), control_levels.max()
        coef_range = max_coeff - min_coeff if max_coeff != min_coeff else 1.0
        rescaled_levels = (control_levels - min_coeff) / coef_range
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        
        for label in choice_labels:
            if legend_options and label in legend_options:
                plot_label = f"Prob({label}) - {legend_options[label][:40]}..."
            else:
                plot_label = f"Prob({label})"
            
            # This is the raw average probability, NOT rescaled across the x-axis
            probs = [results[c].get(label, 0) for c in control_levels]
            # rescale the prob so the sum is 1.0
            if sum(probs) > 0:
                probs = [p / sum(probs) for p in probs]
            else:
                probs = [0.0] * len(probs)
            ax.plot(rescaled_levels, probs, marker='o', linestyle='-', label=plot_label)

        if min_coeff <= 0 <= max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            ax.axvline(x=zero_rescaled_pos, color='grey', linestyle='--', linewidth=1.2)
            ax.text(zero_rescaled_pos, ax.get_ylim()[1] * 0.95, '  No Control', rotation=90, verticalalignment='top', color='grey', fontsize=10)

        ax.set_xlabel(xlabel + " (Rescaled from Actual Coefficients)")
        tick_positions_rescaled = [0.0, 1.0]
        tick_labels_actual = [f"{min_coeff:.2f}", f"{max_coeff:.2f}"]
        if min_coeff < 0 < max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            if 0.05 < zero_rescaled_pos < 0.95:
                tick_positions_rescaled.append(zero_rescaled_pos)
                tick_labels_actual.append("0.0")
        sorted_ticks = sorted(zip(tick_positions_rescaled, tick_labels_actual))
        final_tick_positions, final_tick_labels = zip(*sorted_ticks)
        ax.set_xticks(final_tick_positions)
        ax.set_xticklabels(final_tick_labels)
        ax.set_title(plot_title, fontsize=14)
        ax.set_ylabel("Average Probability of Choice")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        model_plot_dir = os.path.join(self.plot_dir, model_prefix)
        os.makedirs(model_plot_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_permuted.png"
        plt.savefig(os.path.join(model_plot_dir, filename))
        plt.close(fig)
        print(f"Saved AVERAGE {control_type}{self.experiment_suffix} probability plot for '{mcq_scenario_key}'.")

    # --- NEW SCATTER PLOT METHODS ---

    def _plot_results_scatter(self, detailed_results, avg_probs, control_levels, choice_labels, scenario_name, control_type, plot_title, xlabel, legend_options=None, mcq_scenario_key=""):
        """Plots a scatter of individual scenario probabilities with the average line overlaid."""
        fig, ax = plt.subplots(figsize=(14, 8))
        control_levels = np.array(control_levels)
        min_coeff, max_coeff = control_levels.min(), control_levels.max()
        coef_range = max_coeff - min_coeff if max_coeff != min_coeff else 1.0
        
        # Prepare data for scatter plot
        scatter_data = {label: {'x': [], 'y': []} for label in choice_labels}
        for result in detailed_results:
            level = result['control_level']
            rescaled_level = (level - min_coeff) / coef_range
            for label, prob in result['choice_probabilities'].items():
                if label in scatter_data:
                    scatter_data[label]['x'].append(rescaled_level)
                    scatter_data[label]['y'].append(prob)

        # Plot scatter and average line for each choice
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(choice_labels)))
        for i, label in enumerate(choice_labels):
            # Scatter plot for individual points
            ax.scatter(scatter_data[label]['x'], scatter_data[label]['y'], alpha=0.2, color=colors[i], label=f'Individual Prob({label})')
            
            # Overlay the average line
            rescaled_avg_x = (np.array(list(avg_probs.keys())) - min_coeff) / coef_range
            avg_y = [avg_probs[c].get(label, 0) for c in avg_probs.keys()]
            ax.plot(rescaled_avg_x, avg_y, color=colors[i], marker='o', linestyle='-', label=f'Average Prob({label})')

        # --- Formatting (same as _plot_results) ---
        ax.set_title(plot_title.replace("Prob of", "Prob of (Scatter)"), fontsize=14)
        if min_coeff <= 0 <= max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            ax.axvline(x=zero_rescaled_pos, color='grey', linestyle='--', linewidth=1.2)
        ax.set_xlabel(xlabel + " (Rescaled from Actual Coefficients)")
        # ... (copying the rest of the axis setup)
        tick_positions_rescaled = [0.0, 1.0]
        tick_labels_actual = [f"{min_coeff:.2f}", f"{max_coeff:.2f}"]
        if min_coeff < 0 < max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            if 0.05 < zero_rescaled_pos < 0.95:
                tick_positions_rescaled.append(zero_rescaled_pos)
                tick_labels_actual.append("0.0")
        sorted_ticks = sorted(zip(tick_positions_rescaled, tick_labels_actual))
        final_tick_positions, final_tick_labels = zip(*sorted_ticks)
        ax.set_xticks(final_tick_positions)
        ax.set_xticklabels(final_tick_labels)
        ax.set_ylabel("Probability of Choice")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        model_plot_dir = os.path.join(self.plot_dir, model_prefix)
        os.makedirs(model_plot_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_permuted_scatter.png"
        plt.savefig(os.path.join(model_plot_dir, filename))
        plt.close(fig)
        print(f"Saved SCATTER {control_type} probability plot for '{mcq_scenario_key}'.")


    def _plot_likert_score_scatter(self, detailed_results, control_levels, choice_labels, scenario_name, control_type, mcq_scenario_key=""):
        """Plots a scatter of individual scenario Likert scores with the average line overlaid."""
        fig, ax = plt.subplots(figsize=(12, 8))
        control_levels = np.array(control_levels)
        min_coeff, max_coeff = control_levels.min(), control_levels.max()
        coef_range = max_coeff - min_coeff if max_coeff != min_coeff else 1.0
        
        likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}
        
        # Prepare data for scatter plot
        scatter_x, scatter_y = [], []
        for result in detailed_results:
            level = result['control_level']
            rescaled_level = (level - min_coeff) / coef_range
            score = self._calculate_likert_from_probs(result['choice_probabilities'], likert_weights)
            scatter_x.append(rescaled_level)
            scatter_y.append(score)
            
        ax.scatter(scatter_x, scatter_y, alpha=0.2, color='darkorange', label='Individual Scenario Score')

        # Calculate and overlay the average line
        avg_scores_by_level = {level: [] for level in control_levels}
        for x, y in zip(scatter_x, scatter_y):
            # Convert rescaled x back to original level to group correctly
            original_level = x * coef_range + min_coeff
            # Find the closest level in control_levels to handle potential float precision issues
            closest_level = min(control_levels, key=lambda c: abs(c - original_level))
            avg_scores_by_level[closest_level].append(y)
            
        avg_x_rescaled = (np.array(sorted(avg_scores_by_level.keys())) - min_coeff) / coef_range
        avg_y_scores = [np.mean(avg_scores_by_level[level]) for level in sorted(avg_scores_by_level.keys())]
        
        ax.plot(avg_x_rescaled, avg_y_scores, marker='o', linestyle='-', color='darkred', label='Average Likert Score')

        # --- Formatting ---
        ax.set_title(f"Average Likert Score (Scatter) vs. {control_type.replace('_', ' ')} Control\n({scenario_name} - MCQ: {mcq_scenario_key})", fontsize=14)
        if min_coeff <= 0 <= max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            ax.axvline(x=zero_rescaled_pos, color='grey', linestyle='--', linewidth=1.2)
        ax.set_xlabel(f"Control Level ({control_type.replace('_', ' ')}, Rescaled)")
        # ... (copying the rest of the axis setup)
        tick_positions_rescaled = [0.0, 1.0]
        tick_labels_actual = [f"{min_coeff:.2f}", f"{max_coeff:.2f}"]
        if min_coeff < 0 < max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            if 0.05 < zero_rescaled_pos < 0.95:
                tick_positions_rescaled.append(zero_rescaled_pos)
                tick_labels_actual.append("0.0")
        sorted_ticks = sorted(zip(tick_positions_rescaled, tick_labels_actual))
        final_tick_positions, final_tick_labels = zip(*sorted_ticks)
        ax.set_xticks(final_tick_positions)
        ax.set_xticklabels(final_tick_labels)
        ax.set_ylabel("Average Likert Score (4=A, 0=E)")
        ax.set_ylim(0, 4)
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        model_plot_dir = os.path.join(self.plot_dir, model_prefix)
        os.makedirs(model_plot_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_likert_scatter.png"
        plt.savefig(os.path.join(model_plot_dir, filename))
        plt.close(fig)
        print(f"Saved SCATTER Likert score plot for '{mcq_scenario_key}'.")
        
    def _plot_probability_sum(self, results, control_levels, choice_labels, scenario_name, control_type, mcq_scenario_key=""):
        """Plots the sum of probabilities for all choices vs. rescaled control level."""
        plt.figure(figsize=(12, 8))
        min_coeff = float(min(control_levels))
        max_coeff = float(max(control_levels))
        coef_range = max_coeff - min_coeff if max_coeff != min_coeff else 1.0
        rescaled_levels = [(c - min_coeff) / coef_range for c in control_levels]
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        total_probs = []
        for c in control_levels:
            level_probs = results.get(c, {})
            total_prob = sum(level_probs.get(label, 0) for label in choice_labels)
            total_probs.append(total_prob)
        plt.plot(rescaled_levels, total_probs, marker='o', linestyle='-', label='Sum of Probabilities (A-E)')
        plt.title(f"Sum of Choice Probabilities vs. {control_type} Control\n({scenario_name} - MCQ: {mcq_scenario_key})")
        plt.xlabel(f"Control Level ({control_type}, Rescaled 0-1)")
        plt.ylabel("Sum of Probabilities")
        plt.ylim(0, 1.1)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        model_plot_dir = os.path.join(self.plot_dir, model_prefix)
        os.makedirs(model_plot_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_prob_sum.png"
        plt.savefig(os.path.join(model_plot_dir, filename))
        plt.close()
        print(f"Saved probability sum plot to {filename}")

    # In ControlExperiment class in base.py

    def _plot_likert_score(self, results, control_levels, choice_labels, scenario_name, control_type, mcq_scenario_key=""):
        """
        Plots the average Likert score vs. rescaled control level.
        IMPROVED: Now marks the zero-control point and has smarter x-axis labels.
        """
        fig, ax = plt.subplots(figsize=(12, 8)) # Use fig, ax for more control

        control_levels = np.array(control_levels)
        min_coeff, max_coeff = control_levels.min(), control_levels.max()
        coef_range = max_coeff - min_coeff if max_coeff != min_coeff else 1.0
        rescaled_levels = (control_levels - min_coeff) / coef_range

        # Use os.path.basename to get a clean model name for the prefix
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        
        # --- Calculate Likert Scores ---
        likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}
        likert_scores = []
        for c in control_levels:
            level_probs = results.get(c, {})
            total_prob = sum(level_probs.get(label, 0) for label in choice_labels)
            # Normalize probabilities for the choices so they sum to 1 for a stable score
            rescaled_level_probs = {label: (level_probs.get(label, 0) / total_prob) if total_prob > 0 else 0 for label in choice_labels}
            score = sum(rescaled_level_probs.get(label, 0) * likert_weights.get(label, 0) for label in choice_labels)
            likert_scores.append(score)
            
        ax.plot(rescaled_levels, likert_scores, marker='o', linestyle='-', label='Average Likert Score', color='green')

        # --- PLOTTING IMPROVEMENTS ---
        # Mark the zero-control (baseline) point
        if min_coeff <= 0 <= max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            ax.axvline(x=zero_rescaled_pos, color='grey', linestyle='--', linewidth=1.2)
            ax.text(zero_rescaled_pos, ax.get_ylim()[1] * 0.95, '  No Control', rotation=90, 
                    verticalalignment='top', color='grey', fontsize=10)

        # Create smarter X-axis ticks
        ax.set_xlabel(f"Control Level ({control_type.replace('_', ' ')}, Rescaled from Actual Coefficients)")
        tick_positions_rescaled = [0.0, 1.0]
        tick_labels_actual = [f"{min_coeff:.2f}", f"{max_coeff:.2f}"]
        if min_coeff < 0 < max_coeff:
            zero_rescaled_pos = (0 - min_coeff) / coef_range
            if 0.05 < zero_rescaled_pos < 0.95:
                tick_positions_rescaled.append(zero_rescaled_pos)
                tick_labels_actual.append("0.0")
        sorted_ticks = sorted(zip(tick_positions_rescaled, tick_labels_actual))
        final_tick_positions, final_tick_labels = zip(*sorted_ticks)
        ax.set_xticks(final_tick_positions)
        ax.set_xticklabels(final_tick_labels)

        # --- Final Touches ---
        ax.set_title(f"Average Likert Score vs. {control_type.replace('_', ' ')} Control\n({scenario_name} - MCQ: {mcq_scenario_key})", fontsize=14)
        ax.set_ylabel("Average Likert Score (4=A, 0=E)")
        ax.set_ylim(0, 4)
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        
        model_plot_dir = os.path.join(self.plot_dir, model_prefix)
        os.makedirs(model_plot_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_likert.png"
        plt.savefig(os.path.join(model_plot_dir, filename))
        plt.close(fig)
        print(f"Saved Likert score plot to {filename}")

    def save_plot_data(self, results, control_levels, choice_labels, scenario_name, control_type, mcq_scenario_key):
        """Saves the data used for plotting to a JSON file, with rescaled control levels and mapping info."""
        min_coeff = float(min(control_levels))
        max_coeff = float(max(control_levels))
        coef_range = max_coeff - min_coeff if max_coeff != min_coeff else 1.0
        rescaled_levels = [(c - min_coeff) / coef_range for c in control_levels]
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        # Data for _plot_results
        results_data = {label: [results[c].get(label, 0) for c in control_levels] for label in choice_labels}
        # Data for _plot_probability_sum
        total_probs = []
        for c in control_levels:
            level_probs = results.get(c, {})
            total_prob = sum(level_probs.get(label, 0) for label in choice_labels)
            total_probs.append(total_prob)
        # Data for _plot_likert_score
        likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}
        likert_scores = []
        for c in control_levels:
            level_probs = results.get(c, {})
            total_prob = sum(level_probs.get(label, 0) for label in choice_labels)
            rescaled_level_probs = {label: (level_probs.get(label, 0) / total_prob) if total_prob > 0 else 0 for label in choice_labels}
            score = sum(rescaled_level_probs.get(label, 0) * likert_weights.get(label, 0) for label in choice_labels)
            likert_scores.append(score)
        plot_data = {
            'control_levels_rescaled': rescaled_levels,
            'control_levels_actual': list(control_levels),
            'choice_labels': choice_labels,
            'results_by_choice': results_data,
            'probability_sum': total_probs,
            'likert_scores': likert_scores,
            'control_coef_mapping': {
                'base': min_coeff,
                'coef': coef_range,
                'formula': 'real_control = rescaled * coef + base',
                'real_range': [min_coeff, max_coeff]
            }
        }
        results_dir = os.path.join("results", model_prefix)
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_plot_data.json"
        file_path = os.path.join(results_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, indent=4, ensure_ascii=False)
        print(f"Saved plot data to {file_path}")

    def save_results_to_json(self, results_list, scenario_name, control_type, mcq_scenario_key):
        """Saves a list of detailed results to a JSON file."""
        model_prefix = os.path.basename(self.model.config.name_or_path) if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path') else 'model'
        results_dir = os.path.join("results", model_prefix)
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{model_prefix}_{scenario_name.lower().replace(' ', '_')}_{mcq_scenario_key}_vs_{control_type}{self.experiment_suffix}_detailed_results.json"
        file_path = os.path.join(results_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=4, ensure_ascii=False)
        print(f"Saved detailed results to {file_path}")