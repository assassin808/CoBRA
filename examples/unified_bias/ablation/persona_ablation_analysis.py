#!/usr/bin/env python3
"""
Persona Ablation Analysis Script
Analyzes the effect of different personas across control methods for bias experiments.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Import metrics from evaluate_metrics.py
import sys
# Add control directory to path (relative to this script's location)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'control'))

def load_personas(persona_file):
    """Load personas from JSON file and select 10 diverse ones."""
    with open(persona_file, 'r') as f:
        all_personas = json.load(f)
    
    # Select 10 diverse personas (you can modify this selection)
#     echo "Selected personas for ablation:"
# echo "1. Adam Smith - philosopher"  
# echo "2. Giorgio Rossi - mathematician"
# echo "3. Ayesha Khan - college student/literature"
# echo "4. Carlos Gomez - poet"
# echo "5. Francisco Lopez - actor/comedian"
# echo "6. Abigail Chen - digital artist"
# echo "7. Arthur Burton - bartender/bar owner"
# echo "8. Carmen Ortiz - shopkeeper"
# echo "9. Hailey Johnson - writer"
# echo "10. Isabella Rodriguez - cafe owner"
# echo ""
    selected_personas = [
        "Adam Smith",       # Philosopher
        "Giorgio Rossi",    # Mathematician  
        "Ayesha Khan",      # Literature student
        "Carlos Gomez",     # Poet
        "Francisco Lopez",  # Actor/comedian
        "Arthur Burton",    # Bartender
        "Carmen Ortiz",     # Shopkeeper
        "Abigail Chen",     # Artist (painter)
        "Hailey Johnson",   # Writer
        "Isabella Rodriguez"  # cafe owner
    ]
    
    personas = {}
    for name in selected_personas:
        if name in all_personas:
            personas[name] = all_personas[name]
        else:
            print(f"⚠️  Warning: Persona '{name}' not found in file")
    
    return personas

def extract_metrics_from_data(plot_data_file):
    """Extract metrics from a plot data file with NaN filtering."""
    try:
        from evaluate_metrics import compute_monotonicity, compute_smoothness, compute_control_efficacy
    except ImportError:
        print("❌ Cannot import metrics functions. Using basic calculations.")
        return None
    
    with open(plot_data_file, 'r') as f:
        data = json.load(f)
    
    control_levels = data['control_levels_actual']
    likert_scores = data['likert_scores']
    
    # Filter out NaN values and zeros that indicate failed computations
    filtered_control = []
    filtered_likert = []
    
    for ctrl, score in zip(control_levels, likert_scores):
        if score != 0 and str(score).lower() != 'nan' and not (isinstance(score, float) and np.isnan(score)):
            filtered_control.append(ctrl)
            filtered_likert.append(score)
    
    if len(filtered_likert) < 2:
        return {
            'ndcg': float('nan'),
            'rho': float('nan'),
            'smoothness_1': float('nan'),
            'smoothness_2': float('nan'),
            'efficacy': float('nan'),
            'valid_points': len(filtered_likert),
            'total_points': len(likert_scores)
        }
    
    # Convert to numpy arrays to ensure proper type handling
    try:
        filtered_control = np.array(filtered_control, dtype=np.float64)
        filtered_likert = np.array(filtered_likert, dtype=np.float64)
        
        # Calculate metrics
        ndcg_score, correlation = compute_monotonicity(filtered_control, filtered_likert)
        smoothness_1, smoothness_2 = compute_smoothness(filtered_likert)
        efficacy = compute_control_efficacy(filtered_likert)
        
        return {
            'ndcg': float(ndcg_score) if not np.isnan(ndcg_score) else float('nan'),
            'rho': float(correlation) if not np.isnan(correlation) else float('nan'),
            'smoothness_1': float(smoothness_1) if not np.isnan(smoothness_1) else float('nan'),
            'smoothness_2': float(smoothness_2) if not np.isnan(smoothness_2) else float('nan'),
            'efficacy': float(efficacy) if not np.isnan(efficacy) else float('nan'),
            'valid_points': len(filtered_likert),
            'total_points': len(likert_scores)
        }
    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        return {
            'ndcg': float('nan'),
            'rho': float('nan'),
            'smoothness_1': float('nan'),
            'smoothness_2': float('nan'),
            'efficacy': float('nan'),
            'valid_points': len(filtered_likert),
            'total_points': len(likert_scores)
        }

def collect_persona_data(results_dir, personas):
    """Collect data across all personas for different methods."""
    # Define the methods and scenarios we want to analyze
    methods = {
        'prompt_likert': 'Prompt Control',
        'repe_linear_comb': 'RepE Linear',
        'repe_orthognalize': 'RepE Orthogonal',
        'repe_projection': 'RepE Projection'
    }
    
    scenarios = ['asian', 'invest', 'hotel', 'asch']
    
    results = defaultdict(lambda: defaultdict(dict))
    plot_data = defaultdict(lambda: defaultdict(dict))
    
    # Check if results_dir exists, if not try to find the actual results directory
    if not os.path.exists(results_dir):
        # Try looking for results in common locations
        possible_dirs = [
            './results/mistral-7B-Instruct-v0.3',
            '../results/mistral-7B-Instruct-v0.3',
            '../../results/mistral-7B-Instruct-v0.3'
        ]
        
        for possible_dir in possible_dirs:
            if os.path.exists(possible_dir):
                results_dir = possible_dir
                print(f"Found results directory: {results_dir}")
                break
        else:
            print(f"❌ Results directory not found: {results_dir}")
            return results, plot_data
    
    model_name = "mistral-7B-Instruct-v0.3"  # Fixed model name
    
    for scenario in scenarios:
        for method_key, method_name in methods.items():
            for persona_name in personas.keys():
                # Convert persona name to file-safe format (replace spaces with underscores, lowercase)
                persona_file_name = persona_name.replace(' ', '_')
                persona_lower = persona_name.lower().replace(' ', '_')
                
                # Look for files with the actual naming pattern
                # Pattern: mistral-7B-Instruct-v0.3_milgram_(persona:_adam_smith)_choice_first_vs_repe_projection_persona_Adam_Smith_plot_data.json
                filename_patterns = [
                    f"{model_name}_{scenario}_(persona:_{persona_lower})_choice_first_vs_{method_key}_persona_{persona_file_name}_plot_data.json",
                    f"{model_name}_{scenario}_(persona:_{persona_lower})_choice_first_vs_{method_key}_plot_data.json",
                    f"{model_name}_{scenario}_choice_first_vs_{method_key}_persona_{persona_file_name}_plot_data.json"
                ]
                
                found_file = None
                for pattern in filename_patterns:
                    filepath = os.path.join(results_dir, pattern)
                    if os.path.exists(filepath):
                        found_file = filepath
                        break
                
                if found_file:
                    try:
                        # Extract metrics
                        metrics = extract_metrics_from_data(found_file)
                        if metrics:
                            results[scenario][method_name][persona_name] = metrics
                            
                            # Store plot data for visualization
                            with open(found_file, 'r') as f:
                                data = json.load(f)
                            
                            # Filter out NaN and zero values for plotting
                            control_levels = data['control_levels_actual']
                            likert_scores = data['likert_scores']
                            
                            # Create filtered data for plotting
                            filtered_control = []
                            filtered_likert = []
                            
                            for ctrl, score in zip(control_levels, likert_scores):
                                if score != 0 and str(score).lower() != 'nan':
                                    filtered_control.append(ctrl)
                                    filtered_likert.append(score)
                            
                            plot_data[scenario][method_name][persona_name] = {
                                'control_levels': filtered_control,
                                'likert_scores': filtered_likert
                            }
                            
                            print(f"✓ Loaded: {scenario} - {method_name} - {persona_name}")
                            if len(filtered_likert) != len(likert_scores):
                                print(f"    ⚠️  Filtered out {len(likert_scores) - len(filtered_likert)} invalid data points")
                    except Exception as e:
                        print(f"✗ Error loading {found_file}: {e}")
                else:
                    print(f"✗ Missing: {scenario} - {method_name} - {persona_name}")
    
    return results, plot_data

def create_persona_comparison_plots(plot_data, personas, output_dir):
    """Create temperature-ablation-style plots where each persona is a different line within each method."""
    print("📊 Creating persona comparison plots (all personas per method)...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = list(plot_data.keys())
    if not scenarios:
        print("❌ No plot data available")
        return
    
    methods = list(plot_data[scenarios[0]].keys()) if scenarios else []
    personas_list = list(personas.keys())
    
    # Color scheme for personas - use a colormap with enough distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(personas_list), 10)))
    if len(personas_list) > 10:
        # Use additional colormap for more personas
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(personas_list) - 10))
        colors = np.vstack([colors, colors2])
    
    persona_colors = {persona: colors[i] for i, persona in enumerate(personas_list)}
    
    for scenario in scenarios:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Persona Comparison Analysis: {scenario.replace("_", " ").title()}', 
                    fontsize=18, fontweight='bold')
        
        # Collect all y-values for this scenario to determine global y-axis range
        all_y_values = []
        method_y_ranges = {}
        
        for method in methods:
            method_y_values = []
            for persona in personas_list:
                if persona in plot_data[scenario][method]:
                    data = plot_data[scenario][method][persona]
                    likert_scores = np.array(data['likert_scores'])
                    method_y_values.extend(likert_scores)
                    all_y_values.extend(likert_scores)
            method_y_ranges[method] = method_y_values
        
        # Calculate global y-range for the scenario
        if all_y_values:
            global_y_min = min(all_y_values)
            global_y_max = max(all_y_values)
            global_y_range = global_y_max - global_y_min
        else:
            global_y_min, global_y_max, global_y_range = 0, 4, 4
        
        for idx, method in enumerate(methods):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Calculate method-specific y-range
            method_y_values = method_y_ranges.get(method, [])
            if method_y_values:
                method_y_min = min(method_y_values)
                method_y_max = max(method_y_values)
                method_y_range = method_y_max - method_y_min
            else:
                method_y_min, method_y_max, method_y_range = global_y_min, global_y_max, global_y_range
            
            persona_count = 0
            # Plot all personas for this method
            for persona in personas_list:
                if persona in plot_data[scenario][method]:
                    data = plot_data[scenario][method][persona]
                    control_levels = np.array(data['control_levels'])
                    likert_scores = np.array(data['likert_scores'])
                    
                    # Skip if no valid data points
                    if len(control_levels) == 0:
                        continue
                    
                    # Normalize control levels properly, aligning 0.0 control across methods
                    # Find the 0.0 control point or closest to it
                    zero_control_idx = np.argmin(np.abs(control_levels))
                    zero_control_value = control_levels[zero_control_idx]
                    
                    # Create symmetric range around zero control
                    control_range = max(np.abs(control_levels - zero_control_value))
                    if control_range > 0:
                        # Normalize to [-1, 1] range centered on zero control, then shift to [0, 1]
                        control_normalized = (control_levels - zero_control_value) / control_range
                        control_normalized = (control_normalized + 1) / 2  # Shift to [0, 1]
                    else:
                        control_normalized = np.array([0.5])  # Single point goes to middle
                    
                    ax.plot(control_normalized, likert_scores, 
                           color=persona_colors[persona], alpha=0.8, linewidth=2.5,
                           label=persona, marker='o', markersize=5)
                    persona_count += 1
            
            # Smart y-axis scaling
            if method_y_range < 0.5:  # If the range is very small, zoom in
                # Add padding around the data
                padding = max(0.1, method_y_range * 0.2)
                ax.set_ylim(method_y_min - padding, method_y_max + padding)
                ax.set_title(f'{method} (Zoomed: Range={method_y_range:.3f})', fontsize=14, fontweight='bold')
            elif global_y_range > 2.0:  # If global range is large, use global range
                padding = global_y_range * 0.05
                ax.set_ylim(global_y_min - padding, global_y_max + padding)
                ax.set_title(f'{method}', fontsize=14, fontweight='bold')
            else:  # Standard scaling
                ax.set_ylim(0, 4.2)  # Standard Likert scale with small padding
                ax.set_title(f'{method}', fontsize=14, fontweight='bold')
            
            ax.set_xlabel('Control Level (Normalized, 0.5 = Zero Control)', fontsize=12)
            ax.set_ylabel('Likert Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add vertical line at 0.5 to indicate zero control
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add persona count annotation
            ax.text(0.02, 0.98, f'Personas: {persona_count}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # Add range annotation for zoomed plots
            if method_y_range < 0.5:
                ax.text(0.02, 0.88, f'Y-range: [{method_y_min:.2f}, {method_y_max:.2f}]', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Create a single legend for all personas outside the subplots
        handles, labels = [], []
        for persona in personas_list:
            if any(persona in plot_data[scenario][method] for method in methods):
                handles.append(plt.Line2D([0], [0], color=persona_colors[persona], linewidth=2.5, label=persona))
                labels.append(persona)
        
        # Place legend to the right of the subplots
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), 
                  fontsize=11, title="Personas", title_fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend
        plt.savefig(os.path.join(output_dir, f'{scenario}_persona_comparison_by_method.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved persona comparison plot: {scenario}_persona_comparison_by_method.png")

def create_persona_plots(plot_data, personas, output_dir):
    """Create comprehensive persona ablation plots."""
    print("📊 Creating persona ablation plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = list(plot_data.keys())
    if not scenarios:
        print("❌ No plot data available")
        return
    
    methods = list(plot_data[scenarios[0]].keys()) if scenarios else []
    personas_list = list(personas.keys())
    
    # Color scheme for personas
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas_list)))
    persona_colors = {persona: colors[i] for i, persona in enumerate(personas_list)}
    
    # 1. Create overview plots for each scenario
    for scenario in scenarios:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Persona Ablation Analysis: {scenario.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # Collect all y-values for this scenario to determine global y-axis range
        all_y_values = []
        method_y_ranges = {}
        
        for method in methods:
            method_y_values = []
            for persona_name in personas_list:
                if persona_name in plot_data[scenario][method]:
                    data = plot_data[scenario][method][persona_name]
                    likert_scores = np.array(data['likert_scores'])
                    method_y_values.extend(likert_scores)
                    all_y_values.extend(likert_scores)
            method_y_ranges[method] = method_y_values
        
        # Calculate global y-range for the scenario
        if all_y_values:
            global_y_min = min(all_y_values)
            global_y_max = max(all_y_values)
            global_y_range = global_y_max - global_y_min
        else:
            global_y_min, global_y_max, global_y_range = 0, 4, 4
        
        for idx, method in enumerate(methods):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Calculate method-specific y-range
            method_y_values = method_y_ranges.get(method, [])
            if method_y_values:
                method_y_min = min(method_y_values)
                method_y_max = max(method_y_values)
                method_y_range = method_y_max - method_y_min
            else:
                method_y_min, method_y_max, method_y_range = global_y_min, global_y_max, global_y_range
            
            # Plot all personas for this method
            for persona_name in personas_list:
                if persona_name in plot_data[scenario][method]:
                    data = plot_data[scenario][method][persona_name]
                    control_levels = np.array(data['control_levels'])
                    likert_scores = np.array(data['likert_scores'])
                    
                    # Skip if no valid data points
                    if len(control_levels) == 0:
                        continue
                    
                    # Normalize control levels properly, aligning 0.0 control across methods
                    zero_control_idx = np.argmin(np.abs(control_levels))
                    zero_control_value = control_levels[zero_control_idx]
                    
                    # Create symmetric range around zero control
                    control_range = max(np.abs(control_levels - zero_control_value))
                    if control_range > 0:
                        # Normalize to [-1, 1] range centered on zero control, then shift to [0, 1]
                        control_normalized = (control_levels - zero_control_value) / control_range
                        control_normalized = (control_normalized + 1) / 2  # Shift to [0, 1]
                    else:
                        control_normalized = np.array([0.5])  # Single point goes to middle
                    
                    ax.plot(control_normalized, likert_scores, 
                           color=persona_colors[persona_name], alpha=0.8, linewidth=2,
                           label=persona_name, marker='o', markersize=4)
            
            # Smart y-axis scaling
            if method_y_range < 0.5:  # If the range is very small, zoom in
                padding = max(0.1, method_y_range * 0.2)
                ax.set_ylim(method_y_min - padding, method_y_max + padding)
                ax.set_title(f'{method} (Zoomed: Range={method_y_range:.3f})', fontsize=14, fontweight='bold')
            elif global_y_range > 2.0:  # If global range is large, use global range
                padding = global_y_range * 0.05
                ax.set_ylim(global_y_min - padding, global_y_max + padding)
                ax.set_title(f'{method}', fontsize=14, fontweight='bold')
            else:  # Standard scaling
                ax.set_ylim(0, 4.2)  # Standard Likert scale with small padding
                ax.set_title(f'{method}', fontsize=14, fontweight='bold')
            
            ax.set_xlabel('Control Level (Normalized, 0.5 = Zero Control)', fontsize=12)
            ax.set_ylabel('Likert Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add vertical line at 0.5 to indicate zero control
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Create a single legend for all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{scenario}_persona_ablation_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved overview plot: {scenario}_persona_ablation_overview.png")

    # 2. Create individual zoomed plots for each method
    create_zoomed_individual_persona_plots(plot_data, output_dir, persona_colors, personas_list, methods)

def create_zoomed_individual_persona_plots(plot_data, output_dir, persona_colors, personas_list, methods):
    """Create individual zoomed plots for each method to better show persona variations."""
    scenarios = list(plot_data.keys())
    
    for scenario in scenarios:
        for method in methods:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Collect all y-values for this method
            method_y_values = []
            valid_personas = []
            
            for persona_name in personas_list:
                if persona_name in plot_data[scenario][method]:
                    data = plot_data[scenario][method][persona_name]
                    likert_scores = np.array(data['likert_scores'])
                    method_y_values.extend(likert_scores)
                    valid_personas.append(persona_name)
            
            if not method_y_values:
                continue
                
            method_y_min = min(method_y_values)
            method_y_max = max(method_y_values)
            method_y_range = method_y_max - method_y_min
            
            # Plot all personas for this method
            for persona_name in valid_personas:
                data = plot_data[scenario][method][persona_name]
                control_levels = np.array(data['control_levels'])
                likert_scores = np.array(data['likert_scores'])
                
                # Skip if no valid data points
                if len(control_levels) == 0:
                    continue
                
                # Normalize control levels
                zero_control_idx = np.argmin(np.abs(control_levels))
                zero_control_value = control_levels[zero_control_idx]
                
                control_range = max(np.abs(control_levels - zero_control_value))
                if control_range > 0:
                    control_normalized = (control_levels - zero_control_value) / control_range
                    control_normalized = (control_normalized + 1) / 2
                else:
                    control_normalized = np.array([0.5])
                
                ax.plot(control_normalized, likert_scores, 
                       color=persona_colors[persona_name], alpha=0.8, linewidth=2.5,
                       label=persona_name, marker='o', markersize=6)
            
            # Smart y-axis scaling
            if method_y_range < 1.0:  # Zoom in for small ranges
                padding = max(0.05, method_y_range * 0.15)
                ax.set_ylim(method_y_min - padding, method_y_max + padding)
                title_suffix = f' (Zoomed - Range: {method_y_range:.3f})'
            else:
                padding = method_y_range * 0.05
                ax.set_ylim(method_y_min - padding, method_y_max + padding)
                title_suffix = f' (Range: {method_y_range:.3f})'
            
            ax.set_title(f'{scenario.replace("_", " ").title()}: {method}{title_suffix}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Control Level (Normalized, 0.5 = Zero Control)', fontsize=12)
            ax.set_ylabel('Likert Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # Add vertical line at 0.5 to indicate zero control
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add statistics box
            stats_text = f'Min: {method_y_min:.3f}\nMax: {method_y_max:.3f}\nRange: {method_y_range:.3f}\nPersonas: {len(valid_personas)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', 
                   facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            safe_method = method.replace(' ', '_').replace('/', '_')
            safe_scenario = scenario.replace(' ', '_').replace('(', '').replace(')', '')
            plt.savefig(os.path.join(output_dir, f'{safe_scenario}_{safe_method}_persona_zoomed.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved zoomed persona plot: {safe_scenario}_{safe_method}_persona_zoomed.png")

def create_persona_metrics_heatmaps(results, output_dir):
    """Create heatmaps showing metrics across personas and methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = list(results.keys())
    
    for scenario in scenarios:
        methods = list(results[scenario].keys())
        all_personas = set()
        for method in methods:
            all_personas.update(results[scenario][method].keys())
        all_personas = sorted(list(all_personas))
        
        metrics = ['ndcg', 'rho', 'smoothness_1', 'smoothness_2', 'efficacy']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Persona Metrics Heatmaps: {scenario.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        for metric_idx, metric in enumerate(metrics):
            row = metric_idx // 3
            col = metric_idx % 3
            ax = axes[row, col]
            
            # Create matrix for heatmap
            data_matrix = np.full((len(methods), len(all_personas)), np.nan)
            
            for i, method in enumerate(methods):
                for j, persona in enumerate(all_personas):
                    if persona in results[scenario][method]:
                        value = results[scenario][method][persona].get(metric, np.nan)
                        if not np.isnan(value):
                            data_matrix[i, j] = value
            
            # Create heatmap
            mask = np.isnan(data_matrix)
            sns.heatmap(data_matrix, 
                       xticklabels=[p[:10] + '...' if len(p) > 10 else p for p in all_personas],
                       yticklabels=methods,
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       mask=mask,
                       ax=ax,
                       cbar_kws={'label': metric.upper()})
            
            ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Persona', fontsize=10)
            ax.set_ylabel('Method', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        if len(metrics) % 3 != 0:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{scenario}_persona_metrics_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved persona metrics heatmap: {scenario}_persona_metrics_heatmap.png")

def create_persona_table(results, personas, output_dir):
    """Create comprehensive persona metrics table."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten results into a table format
    rows = []
    
    for scenario in results:
        for method in results[scenario]:
            for persona_name in results[scenario][method]:
                metrics = results[scenario][method][persona_name]
                row = {
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Method': method,
                    'Persona': persona_name,
                    'NDCG': metrics.get('ndcg', float('nan')),
                    'Spearman ρ': metrics.get('rho', float('nan')),
                    'Smoothness Δ¹': metrics.get('smoothness_1', float('nan')),
                    'Smoothness Δ²': metrics.get('smoothness_2', float('nan')),
                    'Efficacy': metrics.get('efficacy', float('nan')),
                    'Valid Points': metrics.get('valid_points', 0),
                    'Total Points': metrics.get('total_points', 0)
                }
                rows.append(row)
    
    # Save as simple text format (since we don't have pandas)
    csv_path = os.path.join(output_dir, 'persona_ablation_metrics.txt')
    
    with open(csv_path, 'w') as f:
        # Write header
        if rows:
            header = '\t'.join(rows[0].keys())
            f.write(header + '\n')
            
            # Write data rows
            for row in rows:
                values = []
                for value in row.values():
                    if isinstance(value, float):
                        if str(value) == 'nan':
                            values.append('NaN')
                        else:
                            values.append(f'{value:.4f}')
                    else:
                        values.append(str(value))
                f.write('\t'.join(values) + '\n')
    
    print(f"✓ Saved metrics table: {csv_path}")
    
    return rows

def create_persona_summary(results, personas, output_dir):
    """Create summary statistics for persona ablation."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, 'persona_ablation_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("PERSONA ABLATION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Analyze per scenario
        for scenario in results:
            f.write(f"SCENARIO: {scenario}\n")
            f.write("-" * 30 + "\n")
            
            for method in results[scenario]:
                f.write(f"\nMethod: {method}\n")
                
                persona_efficacies = {}
                for persona_name, metrics in results[scenario][method].items():
                    efficacy = metrics.get('efficacy', float('nan'))
                    if str(efficacy) != 'nan':
                        persona_efficacies[persona_name] = efficacy
                
                if persona_efficacies:
                    # Best and worst personas for this method
                    best_persona = max(persona_efficacies, key=persona_efficacies.get)
                    worst_persona = min(persona_efficacies, key=persona_efficacies.get)
                    
                    f.write(f"  Best efficacy: {best_persona} ({persona_efficacies[best_persona]:.3f})\n")
                    f.write(f"  Worst efficacy: {worst_persona} ({persona_efficacies[worst_persona]:.3f})\n")
                    f.write(f"  Range: {max(persona_efficacies.values()) - min(persona_efficacies.values()):.3f}\n")
                    f.write(f"  Valid personas: {len(persona_efficacies)}/{len(personas)}\n")
                else:
                    f.write("  No valid data for any persona\n")
            
            f.write("\n")
    
    print(f"✓ Saved summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Persona ablation analysis for bias control experiments')
    parser.add_argument('--results_dir', 
                       default='./results/mistral-7B-Instruct-v0.3',
                       help='Directory containing the results')
    parser.add_argument('--output_dir', 
                       default='./persona_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--persona_file',
                       default='./personas_extracted.json',
                       help='Path to personas JSON file')
    
    args = parser.parse_args()
    
    print("🎭 PERSONA ABLATION ANALYSIS")
    print("=" * 50)
    
    # Load personas
    print("\n👥 Loading personas...")
    personas = load_personas(args.persona_file)
    
    if not personas:
        print("❌ No personas loaded! Please check the persona file.")
        return
    
    print(f"Selected {len(personas)} personas:")
    for name in personas.keys():
        print(f"  • {name}")
    
    # Collect data
    print("\n📊 Collecting persona data...")
    results, plot_data = collect_persona_data(args.results_dir, personas)
    
    if not results:
        print("❌ No data found! Please check the results directory and file naming patterns.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations and analysis
    print("\n📈 Creating plots...")
    create_persona_comparison_plots(plot_data, personas, args.output_dir)
    create_persona_plots(plot_data, personas, args.output_dir)
    
    print("\n�️ Creating metrics heatmaps...")
    create_persona_metrics_heatmaps(results, args.output_dir)
    
    print("\n�📋 Creating metrics tables...")
    rows = create_persona_table(results, personas, args.output_dir)
    
    print("\n📊 Creating summary...")
    create_persona_summary(results, personas, args.output_dir)
    
    # Print final summary
    print("\n📈 ANALYSIS SUMMARY")
    print("=" * 30)
    print(f"Total combinations analyzed: {len(rows)}")
    print(f"Personas: {list(personas.keys())}")
    
    # Check for data filtering
    filtered_data = [row for row in rows if row['Valid Points'] != row['Total Points']]
    if len(filtered_data) > 0:
        print(f"\n⚠️  DATA FILTERING APPLIED:")
        print(f"   {len(filtered_data)} cases had invalid data points filtered out")
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
