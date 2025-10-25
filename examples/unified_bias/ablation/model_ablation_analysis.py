#!/usr/bin/env python3
"""
Model Ablation Analysis Script
Compares the effect of different models across control methods for bias experiments.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Global font configuration: Comic Sans MS bold (fallback to default bold if missing)
import matplotlib
try:
    plt.rcParams.update({
        'font.family': 'Comic Sans MS',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'legend.fontsize': 'medium'
    })
except Exception:
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

# Import metrics from evaluate_metrics.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'control'))
from evaluate_metrics import compute_monotonicity, compute_smoothness, compute_control_efficacy

# Models to include (exclude gpt-oss)
MODELS = [
    "DeepSeek-R1-0528-Qwen3-8B",
    "Llama-3.1-8B-Instruct",
    "mistral-7B-Instruct-v0.3",
    "Qwen3-8B"
]

# Display abbreviations for legend labels
MODEL_ABBREV = {
    "DeepSeek-R1-0528-Qwen3-8B": "Deepseek",
    "Llama-3.1-8B-Instruct": "Llama",
    "mistral-7B-Instruct-v0.3": "Mistral",
    "Qwen3-8B": "Qwen"
}

def legend_label(model: str) -> str:
    """Return legend label combining abbreviation and full model name for clarity in grouped plots."""
    abbr = MODEL_ABBREV.get(model, model)
    if model == abbr:
        return model
    return f"{abbr} ({model})"

METHODS = {
    'prompt_likert': 'Prompt Numerical',
    'repe_linear_comb': 'RepE Linear',
    'repe_orthognalize': 'RepE Orthogonal',
    'repe_projection': 'RepE Projection',
    # New finetuning (LoRA) method
    'lora': 'Finetuning'
}

SCENARIOS = ['milgram', 'stanford', 'asch', 'hotel', 'bias_info', 'wason','asian', 'invest']

# Bias groups (scenarios plotted together as subplots per method)
BIAS_GROUPS = [
    ('asch', 'hotel'),
    ('milgram', 'stanford'),
    ('asian', 'invest'),
    ('bias_info', 'wason'),
]

def find_plot_data_files(results_dir, model, scenario, method):
    # Find the plot data file for a given model/scenario/method
    pattern = f"{model}_authority_({scenario})_choice_first_vs_{method}_plot_data.json"
    file_path = os.path.join(results_dir, model, pattern)
    if os.path.exists(file_path) and 'baseline' not in pattern:
        return file_path
    # Try alternative patterns if needed
    for fname in os.listdir(os.path.join(results_dir, model)):
        if scenario in fname and method in fname and fname.endswith("plot_data.json") and 'persona' not in fname:
            if 'baseline' in fname or 'detailed_results' in fname:
                continue
            print(f"Found alternative plot data file: {fname}")
            return os.path.join(results_dir, model, fname)
    return None

def extract_metrics_from_data(control_levels, likert_scores):
    """Extract metrics from control levels and likert scores."""
    # Filter out NaN values and zeros that indicate failed computations
    valid_mask = ~(np.isnan(likert_scores) | (likert_scores == 0))
    
    # If we have invalid values, filter them out
    if not np.all(valid_mask):
        control_levels_clean = control_levels[valid_mask]
        likert_scores_clean = likert_scores[valid_mask]
    else:
        control_levels_clean = control_levels
        likert_scores_clean = likert_scores
    
    # Calculate metrics
    ndcg_score, correlation = compute_monotonicity(control_levels_clean, likert_scores_clean)
    smoothness_1, smoothness_2 = compute_smoothness(likert_scores_clean)
    efficacy = compute_control_efficacy(likert_scores_clean)
    
    return {
        'ndcg': ndcg_score,
        'rho': correlation,
        'smoothness_1': smoothness_1,
        'smoothness_2': smoothness_2,
        'efficacy': efficacy,
        'valid_points': len(likert_scores_clean),
        'total_points': len(likert_scores)
    }

def collect_model_data(results_dir):
    # Collect data for all models, scenarios, and methods
    data = {}
    metrics_data = {}
    for scenario in SCENARIOS:
        data[scenario] = {}
        metrics_data[scenario] = {}
        for method_key, method_name in METHODS.items():
            data[scenario][method_name] = {}
            metrics_data[scenario][method_name] = {}
            for model in MODELS:
                plot_file = find_plot_data_files(results_dir, model, scenario, method_key)
                if plot_file:
                    with open(plot_file, 'r') as f:
                        plot_data = json.load(f)
                    control_levels = np.array(plot_data['control_levels_actual'])
                    likert_scores = np.array(plot_data['likert_scores'])
                    # Filter out invalid scores
                    valid = (likert_scores != 0) & (~np.isnan(likert_scores))
                    control_levels = control_levels[valid]
                    likert_scores = likert_scores[valid]
                    data[scenario][method_name][model] = {
                        'control_levels': control_levels,
                        'likert_scores': likert_scores
                    }
                    # Extract metrics
                    metrics = extract_metrics_from_data(control_levels, likert_scores)
                    metrics_data[scenario][method_name][model] = metrics
                else:
                    print(f"Missing: {model} {scenario} {method_name}")
    return data, metrics_data

def plot_model_ablation(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}

    for scenario in data:
        for method in data[scenario]:
            plt.figure(figsize=(10, 7))
            for model in MODELS:
                if model in data[scenario][method]:
                    d = data[scenario][method][model]
                    if len(d['control_levels']) == 0:
                        continue
                    min_val = np.min(d['control_levels'])
                    max_val = np.max(d['control_levels'])
                    if max_val == min_val:
                        control_norm = np.zeros_like(d['control_levels'])
                    else:
                        control_norm = (d['control_levels'] - min_val) / (max_val - min_val)
                    plt.plot(control_norm, d['likert_scores'], label=model, color=model_colors[model], marker='o', markersize=3, linewidth=1.1)
            plt.xlabel('Control Coeficient (rescaled in 0-1)', fontsize=17)
            plt.ylabel(f'Cognitive Bias Index ({scenario})', fontsize=17)
            handles, labels = plt.gca().get_legend_handles_labels()
            new_labels = [MODEL_ABBREV.get(lab, lab) for lab in labels]
            if handles:
                plt.legend(handles, new_labels, fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f"{scenario}_{method.replace(' ', '_')}_model_ablation.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()
            print(f"Saved: {fname}")


def plot_model_ablation_grouped(data, output_dir):
    """Create grouped subplot figures: for each method, plot specified scenario groups side-by-side.

    Each figure corresponds to one method. Within the figure, each bias group (tuple of scenarios)
    produces a row (or combined single row if only one group) with len(group) columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}

    # methods in data dict are display names (values of METHODS mapping)
    method_names = list(next(iter(data.values())).keys()) if data else []
    for method in method_names:
        fig_rows = len(BIAS_GROUPS)
        max_cols = max(len(g) for g in BIAS_GROUPS)
        fig, axes = plt.subplots(fig_rows, max_cols, figsize=(5*max_cols, 4.2*fig_rows), squeeze=False)
        for row_i, group in enumerate(BIAS_GROUPS):
            for col_i in range(max_cols):
                ax = axes[row_i][col_i]
                if col_i >= len(group):
                    ax.axis('off')
                    continue
                scenario = group[col_i]
                if scenario not in data or method not in data[scenario]:
                    ax.axis('off')
                    continue
                for model in MODELS:
                    if model in data[scenario][method]:
                        d = data[scenario][method][model]
                        if len(d['control_levels']) == 0:
                            continue
                        min_val = np.min(d['control_levels'])
                        max_val = np.max(d['control_levels'])
                        control_norm = np.zeros_like(d['control_levels']) if max_val==min_val else (d['control_levels']-min_val)/(max_val-min_val)
                        ax.plot(control_norm, d['likert_scores'], label=MODEL_ABBREV.get(model, model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0)
                ax.set_xlabel('Control Coeficient (rescaled in 0-1)', fontsize=17)
                ax.set_ylabel(f'Cognitive Bias Index ({scenario})', fontsize=17)
                ax.tick_params(labelsize=11)
                ax.grid(True, alpha=0.3)
        handle_map = {}
        for row in axes:
            for ax in row:
                h,l = ax.get_legend_handles_labels()
                for handle,label in zip(h,l):
                    if label not in handle_map:
                        handle_map[label]=handle
        if handle_map:
            fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='upper center', ncol=min(len(handle_map),4), frameon=False, fontsize=19, bbox_to_anchor=(0.5,1.005))
        plt.tight_layout(rect=(0,0,1,0.965))
        slug_method = method.replace(' ', '_').lower()
        out_name = f"grouped_ablation_{slug_method}.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close(fig)
        print(f"Saved grouped figure: {out_name}")


def plot_model_ablation_grouped_alt(data, output_dir):
    """Alternative layout 2x4 (2 rows, 4 columns) flattening groups left-to-right.

    Each scenario gets one subplot; order is the sequence of BIAS_GROUPS flattened.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}
    flat_scenarios = [s for grp in BIAS_GROUPS for s in grp]
    ncols = 4
    nrows = int(np.ceil(len(flat_scenarios)/ncols))
    method_names = list(next(iter(data.values())).keys()) if data else []
    for method in method_names:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 3.8*nrows), squeeze=False)
        for idx, scenario in enumerate(flat_scenarios):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            if scenario not in data or method not in data[scenario]:
                ax.axis('off')
                continue
            for model in MODELS:
                if model in data[scenario][method]:
                    d = data[scenario][method][model]
                    if len(d['control_levels']) == 0:
                        continue
                    min_val = np.min(d['control_levels'])
                    max_val = np.max(d['control_levels'])
                    control_norm = np.zeros_like(d['control_levels']) if max_val==min_val else (d['control_levels']-min_val)/(max_val-min_val)
                    ax.plot(control_norm, d['likert_scores'], label=legend_label(model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0)
            ax.set_xlabel('Control Coeficient (rescaled 0-1)', fontsize=17)
            ax.set_ylabel(f'Cognitive Bias Index ({scenario})', fontsize=17)
            ax.tick_params(labelsize=11)
            ax.grid(True, alpha=0.3)
        total_axes = nrows*ncols
        for j in range(len(flat_scenarios), total_axes):
            rr = j // ncols
            cc = j % ncols
            axes[rr][cc].axis('off')
        handle_map = {}
        for row in axes:
            for ax in row:
                h,l = ax.get_legend_handles_labels()
                for handle,label in zip(h,l):
                    if label not in handle_map:
                        handle_map[label]=handle
        if handle_map:
            fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='upper center', ncol=min(len(handle_map),4), frameon=False, fontsize=19, bbox_to_anchor=(0.5,1.0))
        plt.tight_layout(rect=(0,0,1,0.965))
        slug_method = method.replace(' ', '_').lower()
        out_name = f"grouped_ablation_2x4_{slug_method}.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close(fig)
        print(f"Saved alt grouped figure (2x4): {out_name}")


def plot_model_ablation_per_group_vertical(data, output_dir):
    """Per-group vertical (2x1) plots: each group gets its own figure with two stacked scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}
    method_names = list(next(iter(data.values())).keys()) if data else []
    for method in method_names:
        for group in BIAS_GROUPS:
            fig, axes = plt.subplots(len(group), 1, figsize=(5.2, 3.9*len(group)), squeeze=False)
            for i, scenario in enumerate(group):
                ax = axes[i][0]
                if scenario not in data or method not in data[scenario]:
                    ax.axis('off')
                    continue
                for model in MODELS:
                    if model in data[scenario][method]:
                        d = data[scenario][method][model]
                        if len(d['control_levels']) == 0:
                            continue
                        min_val = np.min(d['control_levels'])
                        max_val = np.max(d['control_levels'])
                        control_norm = np.zeros_like(d['control_levels']) if max_val==min_val else (d['control_levels']-min_val)/(max_val-min_val)
                        ax.plot(control_norm, d['likert_scores'], label=MODEL_ABBREV.get(model, model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0)
                ax.set_xlabel('Control Coeficient (rescaled 0-1)', fontsize=17)
                ax.set_ylabel(f'Cognitive Bias Index ({scenario})', fontsize=17)
                ax.tick_params(labelsize=11)
                ax.grid(True, alpha=0.3)
            handle_map = {}
            for row in axes:
                for ax in row:
                    h,l = ax.get_legend_handles_labels()
                    for handle,label in zip(h,l):
                        if label not in handle_map:
                            handle_map[label]=handle
            if handle_map:
                fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='upper center', ncol=min(len(handle_map),4), frameon=False, fontsize=19, bbox_to_anchor=(0.5,1.005))
            plt.tight_layout(rect=(0,0,1,0.965))
            slug_method = method.replace(' ', '_').lower()
            group_slug = '_'.join(group)
            out_name = f"group_{group_slug}_vertical_{slug_method}.png"
            fig.savefig(os.path.join(output_dir, out_name), dpi=300)
            plt.close(fig)
            print(f"Saved per-group vertical figure: {out_name}")


def plot_model_ablation_per_group_horizontal(data, output_dir):
    """Per-group horizontal (1x2) plots: each group gets its own figure with scenarios side-by-side."""
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}
    method_names = list(next(iter(data.values())).keys()) if data else []
    for method in method_names:
        for group in BIAS_GROUPS:
            ncols = len(group)
            fig, axes = plt.subplots(1, ncols, figsize=(5.2*ncols, 4.0), squeeze=False)
            for j, scenario in enumerate(group):
                ax = axes[0][j]
                if scenario not in data or method not in data[scenario]:
                    ax.axis('off')
                    continue
                for model in MODELS:
                    if model in data[scenario][method]:
                        d = data[scenario][method][model]
                        if len(d['control_levels']) == 0:
                            continue
                        min_val = np.min(d['control_levels'])
                        max_val = np.max(d['control_levels'])
                        control_norm = np.zeros_like(d['control_levels']) if max_val==min_val else (d['control_levels']-min_val)/(max_val-min_val)
                        ax.plot(control_norm, d['likert_scores'], label=MODEL_ABBREV.get(model, model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0)
                ax.set_xlabel('Control Coeficient (rescaled 0-1)', fontsize=17)
                ax.set_ylabel(f'Cognitive Bias Index ({scenario})', fontsize=17)
                ax.tick_params(labelsize=11)
                ax.grid(True, alpha=0.3)
            handle_map = {}
            for row in axes:
                for ax in row:
                    h,l = ax.get_legend_handles_labels()
                    for handle,label in zip(h,l):
                        if label not in handle_map:
                            handle_map[label]=handle
            if handle_map:
                fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='upper center', ncol=min(len(handle_map),4), frameon=False, fontsize=19, bbox_to_anchor=(0.5,1.012))
            plt.tight_layout(rect=(0,0,1,0.965))
            slug_method = method.replace(' ', '_').lower()
            group_slug = '_'.join(group)
            out_name = f"group_{group_slug}_horizontal_{slug_method}.png"
            fig.savefig(os.path.join(output_dir, out_name), dpi=300)
            plt.close(fig)
            print(f"Saved per-group horizontal figure: {out_name}")


def plot_model_ablation_grouped_colwise(data, output_dir):
    """Column-wise grouped layout: 2 rows (scenario position within group), G columns (one per bias group).

    Each column corresponds to a bias group from BIAS_GROUPS. The top row shows the first scenario
    in the group tuple, the bottom row shows the second. This satisfies a layout where each column
    keeps the same bias group together vertically.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}
    ncols = len(BIAS_GROUPS)
    nrows = 2  # because each group has exactly 2 scenarios
    method_names = list(next(iter(data.values())).keys()) if data else []
    for method in method_names:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.0*nrows), squeeze=False)
        for col_i, group in enumerate(BIAS_GROUPS):
            for row_i in range(nrows):
                ax = axes[row_i][col_i]
                if row_i >= len(group):
                    ax.axis('off')
                    continue
                scenario = group[row_i]
                if scenario not in data or method not in data[scenario]:
                    ax.axis('off')
                    continue
                for model in MODELS:
                    if model in data[scenario][method]:
                        d = data[scenario][method][model]
                        if len(d['control_levels']) == 0:
                            continue
                        min_val = np.min(d['control_levels'])
                        max_val = np.max(d['control_levels'])
                        control_norm = np.zeros_like(d['control_levels']) if max_val==min_val else (d['control_levels']-min_val)/(max_val-min_val)
                        ax.plot(control_norm, d['likert_scores'], label=MODEL_ABBREV.get(model, model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0)
                if row_i == nrows - 1:
                    ax.set_xlabel('Control Coeficient (rescaled 0-1)', fontsize=17)
                else:
                    ax.set_xlabel('')
                ax.set_ylabel(f'Cognitive Bias Index ({scenario})', fontsize=17)
                ax.tick_params(labelsize=11)
                ax.grid(True, alpha=0.3)
        handle_map = {}
        for row in axes:
            for ax in row:
                h,l = ax.get_legend_handles_labels()
                for handle,label in zip(h,l):
                    if label not in handle_map:
                        handle_map[label]=handle
        if handle_map:
            fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='upper center', ncol=min(len(handle_map),4), frameon=False, fontsize=19, bbox_to_anchor=(0.5,1.0))
        plt.tight_layout(rect=(0,0,1,0.965))
        slug_method = method.replace(' ', '_').lower()
        out_name = f"grouped_ablation_colwise_{slug_method}.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close(fig)
        print(f"Saved column-wise grouped figure: {out_name}")

def create_metrics_table(metrics_data, output_dir):
    """Create comprehensive metrics table."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten results into a DataFrame
    rows = []
    
    for scenario in metrics_data:
        for method in metrics_data[scenario]:
            for model in metrics_data[scenario][method]:
                metrics = metrics_data[scenario][method][model]
                row = {
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Method': method,
                    'Model': model,
                    'NDCG': metrics.get('ndcg', np.nan),
                    'Spearman ρ': metrics.get('rho', np.nan),
                    'Smoothness Δ¹': metrics.get('smoothness_1', np.nan),
                    'Smoothness Δ²': metrics.get('smoothness_2', np.nan),
                    'Efficacy': metrics.get('efficacy', np.nan),
                    'Valid Points': metrics.get('valid_points', 0),
                    'Total Points': metrics.get('total_points', 0)
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'model_ablation_metrics.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved metrics table: {csv_path}")
    
    # Create summary statistics
    summary_stats = []
    
    for scenario in df['Scenario'].unique():
        scenario_df = df[df['Scenario'] == scenario]
        
        for method in scenario_df['Method'].unique():
            method_df = scenario_df[scenario_df['Method'] == method]
            
            for metric in ['NDCG', 'Spearman ρ', 'Smoothness Δ¹', 'Smoothness Δ²', 'Efficacy']:
                values = method_df[metric].dropna()
                if len(values) > 0:
                    summary_stats.append({
                        'Scenario': scenario,
                        'Method': method,
                        'Metric': metric,
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Best_Model': method_df.loc[values.idxmax(), 'Model'] if metric in ['NDCG', 'Spearman ρ', 'Efficacy'] else method_df.loc[values.idxmin(), 'Model']
                    })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, 'model_ablation_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"Saved summary statistics: {summary_path}")
    
    return df, summary_df

def plot_special_four_panel(data, output_dir):
    """Special 1x4 column layout:
    Column 1: authority (milgram) using repe_linear_comb
    Column 2: bandwagon (asch) using prompt_likert
    Column 3: confirmation (confirmation) using lora (finetuning)
    Column 4: framing (framing) using repe_projection

    We assume scenario tokens available in collected data as:
      authority -> 'milgram' (proxy already in SCENARIOS)
      bandwagon -> 'asch'
      confirmation -> 'confirmation' (may not be in SCENARIOS list; we attempt if present)
      framing -> 'stanford' or 'framing' (choose 'stanford' as existing framing-like scenario if 'framing' absent)
    If some scenario/method pair is missing, that subplot is left blank with a note.
    """
    os.makedirs(output_dir, exist_ok=True)

    # We allow fallback scenario tokens if preferred alias not present in collected data.
    # Column specification: (display_title, [scenario_candidates...], method_key, method_display_override)
    spec = [
        ("Authority", ['milgram','authority'], 'repe_linear_comb', 'RepE Linear'),
        ("Bandwagon", ['asch','bandwagon'], 'prompt_likert', 'Prompt Numerical'),
        # Confirmation bias data lives under 'bias_info' or 'wason' in current SCENARIOS; 'confirmation' may not exist.
        ("Confirmation", ['confirmation','bias_info','wason'], 'lora', 'Finetuning'),
        ("Framing", ['stanford','framing'], 'repe_projection', 'RepE Projection'),
    ]

    # Resolve actual scenarios present in data for each spec entry.
    cols = []
    for title, candidates, method_key, method_disp in spec:
        chosen = None
        for cand in candidates:
            if cand in data:  # data keys are scenarios collected
                chosen = cand
                break
        if chosen is None:
            # keep first candidate to drive a 'No data' panel
            chosen = candidates[0]
        cols.append((title, chosen, method_key, method_disp))

    # Derive colors consistent with other plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}

    fig, axes = plt.subplots(1, len(cols), figsize=(5.2*len(cols), 4.6), squeeze=False)
    axes_row = axes[0]

    for idx, (title, scenario, method_key, method_disp) in enumerate(cols):
        ax = axes_row[idx]
        # Data dict keys: data[scenario][method_display_name][model]
        # Need to translate method_key -> display name used during collection
        method_display_name = METHODS.get(method_key, method_key)
        scenario_present = scenario in data and bool(data[scenario])
        method_present = scenario_present and method_display_name in data.get(scenario, {})
        if not (scenario_present and method_present):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=17, color='gray')
            ax.axis('off')
            continue
        # Plot each model curve
        for model in MODELS:
            if model in data[scenario][method_display_name]:
                d = data[scenario][method_display_name][model]
                if len(d['control_levels']) == 0:
                    continue
                ctrl = d['control_levels']
                likert = d['likert_scores']
                min_val = np.min(ctrl)
                max_val = np.max(ctrl)
                ctrl_norm = np.zeros_like(ctrl) if max_val == min_val else (ctrl - min_val)/(max_val - min_val)
                ax.plot(ctrl_norm, likert, label=MODEL_ABBREV.get(model, model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0, alpha=0.95)
        # Axis formatting per requirements:
        #  - Remove x-axis label text
        #  - Larger y-axis label
        #  - Place bias type + method at bottom of subplot instead of top
        ax.set_xlabel('')
        ax.set_ylabel('Cognitve bais index', fontsize=22)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        # Bottom annotation (inside figure, below plot area). Use transform for axis coords.
        ax.text(0.5, -0.18, f"{title}\n({method_disp})", ha='center', va='top', transform=ax.transAxes, fontsize=19, fontweight='bold')

    # Consolidated legend
    handle_map = {}
    for ax in axes_row:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handle_map:
                handle_map[label] = handle
    if handle_map:
        fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='upper center', ncol=min(len(handle_map), 4), frameon=False, fontsize=22, bbox_to_anchor=(0.5, 1.06))
    # Increase bottom margin to accommodate bottom titles
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_name = 'special_four_panel_ablation.png'
    fig.savefig(os.path.join(output_dir, out_name), dpi=300)
    plt.close(fig)
    print(f"Saved special four-panel figure: {out_name}")

def plot_special_four_panel_2x4(data, output_dir):
    """Extended special layout: 2 rows x 4 columns (column-wise large bias groups).

    Each column corresponds to one high-level bias group and contains the two
    underlying scenario proxies (top = first scenario in pair, bottom = second).

    Column order (groups taken from BIAS_GROUPS constant):
      1. (milgram, stanford)  -> Authority / Framing-like pair  (method: repe_linear_comb)
      2. (asch, hotel)        -> Bandwagon / Social proof      (method: prompt_likert)
      3. (bias_info, wason)   -> Confirmation (information)    (method: lora)
      4. (asian, invest)      -> Framing / Risk framing        (method: repe_projection)

    Fixed method mapping per column mirrors earlier single-row special figure:
        [repe_linear_comb, prompt_likert, lora, repe_projection]

    If a scenario/method pair has no data it is marked "No data". We attempt
    simple fallback aliases (e.g. 'authority' for 'milgram', 'framing' for 'stanford',
    'confirmation' for bias_info/wason) when primary key missing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Column specifications drawn from BIAS_GROUPS (ensure at least 4 groups)
    groups = BIAS_GROUPS[:4]
    method_specs = [
        ('prompt_likert', 'Prompt Numerical'),
        ('repe_linear_comb', 'RepE Linear'),
        ('repe_projection', 'RepE Projection'),
        ('lora', 'Finetuning'),
    ]

    # Possible fallback aliases for scenarios (per scenario)
    fallback_aliases = {
        'milgram': ['milgram', 'authority'],
        'stanford': ['stanford', 'framing'],
        'asch': ['asch', 'bandwagon'],
        'hotel': ['hotel'],
        'bias_info': ['bias_info', 'confirmation'],
        'wason': ['wason', 'confirmation'],
        'asian': ['asian'],
        'invest': ['invest']
    }

    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    model_colors = {model: colors[i] for i, model in enumerate(MODELS)}

    fig, axes = plt.subplots(2, 4, figsize=(4.8*4, 4.8*2), squeeze=False)

    for col_idx, group in enumerate(groups):
        scen_a, scen_b = group
        method_key, method_disp = method_specs[col_idx]
        method_display_name = METHODS.get(method_key, method_key)

        for row_idx, scenario in enumerate([scen_a, scen_b]):
            ax = axes[row_idx][col_idx]

            # Resolve scenario with fallback aliases
            resolved = None
            for cand in fallback_aliases.get(scenario, [scenario]):
                if cand in data and any(data[cand].values()):
                    resolved = cand
                    break
            if resolved is None:
                resolved = scenario  # keep original for messaging

            scenario_present = resolved in data and bool(data[resolved])
            method_present = scenario_present and method_display_name in data.get(resolved, {})

            if not (scenario_present and method_present):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, color='gray')
                title_str = f"{scenario}" if row_idx == 0 else f"{scenario}"
                ax.set_title(title_str, fontsize=19)
                ax.axis('off')
                continue

            # Plot each model curve
            for model in MODELS:
                if model in data[resolved][method_display_name]:
                    d = data[resolved][method_display_name][model]
                    if len(d['control_levels']) == 0:
                        continue
                    ctrl = d['control_levels']
                    likert = d['likert_scores']
                    if len(ctrl) == 0:
                        continue
                    min_val = np.min(ctrl)
                    max_val = np.max(ctrl)
                    ctrl_norm = np.zeros_like(ctrl) if max_val == min_val else (ctrl - min_val)/(max_val - min_val)
                    ax.plot(ctrl_norm, likert, label=MODEL_ABBREV.get(model, model), color=model_colors[model], marker='o', markersize=3, linewidth=1.0, alpha=0.95)

            # Unified axis labels (same for x and y) with larger fonts
            # Remove x-axis text completely
            ax.set_xlabel('')
            if col_idx == 0:
                ax.set_ylabel('Cognitve bais index', fontsize=22)
            ax.tick_params(labelsize=14)
            ax.grid(True, alpha=0.3)
            # Row titles (scenario name) moved to bottom with larger font
            ax.text(0.5, -0.1, f"{scenario}", ha='center', va='top', transform=ax.transAxes, fontsize=18, fontweight='bold')

        # Add column header spanning two rows - properly centered for each column
        # Calculate exact center position for this column's subplot pair
        if col_idx>=2:
            center_x = (col_idx + 0.55) / 4
        else:
            center_x = (col_idx + 0.6) / 4
        fig.text(center_x, 0.91, f"{['Authority','Bandwagon','Confirmation','Framing'][col_idx]}\n({method_disp})", ha='center', va='bottom', fontsize=20, fontweight='bold')

    # Build consolidated legend
    handle_map = {}
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in handle_map:
                    handle_map[label] = handle
    if handle_map:
        fig.legend(list(handle_map.values()), list(handle_map.keys()), loc='lower center', ncol=min(len(handle_map), 6), frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.04))

    # Adjust margins and spacing - reduce h_pad to bring subfigures closer, move legend closer
    plt.tight_layout(rect=(0, 0.08, 1, 0.92), h_pad=0.9, w_pad=0.9)
    out_name = 'special_four_panel_2x4_ablation.png'
    fig.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved special 2x4 four-panel figure: {out_name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Model ablation analysis for bias control experiments')
    parser.add_argument('--results_dir', default='./results', help='Directory containing the results')
    parser.add_argument('--output_dir', default='./model_ablation_plots', help='Output directory for plots')
    parser.add_argument('--special_four', action='store_true', help='Generate special 4-column (authority, bandwagon, confirmation, framing) plot with fixed method per column')
    args = parser.parse_args()

    print("Collecting model data...")
    data, metrics_data = collect_model_data(args.results_dir)
    
    if args.special_four:
        # Fast mode: only plot special four-panel figures
        print("Plotting special four-panel ablation figure...")
        plot_special_four_panel(data, args.output_dir)
        print("Plotting special 2x4 column-wise four-panel ablation figure...")
        plot_special_four_panel_2x4(data, args.output_dir)
    else:
        # Full mode: plot all figures
        print("Plotting model ablation (per-scenario)...")
        plot_model_ablation(data, args.output_dir)
        print("Plotting grouped ablation figures (4x2)...")
        plot_model_ablation_grouped(data, args.output_dir)
        print("Plotting grouped ablation figures (2x4)...")
        plot_model_ablation_grouped_alt(data, args.output_dir)
        print("Plotting grouped ablation figures (colwise 2xG)...")
        plot_model_ablation_grouped_colwise(data, args.output_dir)
        print("Plotting per-group vertical (2x1) figures...")
        plot_model_ablation_per_group_vertical(data, args.output_dir)
        print("Plotting per-group horizontal (1x2) figures...")
        plot_model_ablation_per_group_horizontal(data, args.output_dir)
        print("Creating metrics tables...")
        create_metrics_table(metrics_data, args.output_dir)
        print("Plotting special 2x4 column-wise four-panel ablation figure...")
        plot_special_four_panel_2x4(data, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()