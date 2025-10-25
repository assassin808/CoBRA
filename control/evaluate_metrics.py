# evaluate_metrics.py

import os
import json
import numpy as np
import argparse
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Metrics Functions ---
def compute_monotonicity(control_coeffs, output_values, alpha=1.0):
    """
    Compute monotonicity using NDCG-based approach with global deviation mapping.
    
    This advanced method compares the actual sequence against the ideal monotonic sequence:
    1. Creates ideal monotonic sequence by sorting all y-values
    2. For each position i, compares Y_actual[i] vs Y_ideal[i] 
    3. Maps global deviations to relevance scores using exponential function
    4. Uses NDCG to measure how well actual sequence matches ideal structure
    
    This captures both absolute magnitude and relative position, solving the issue where
    local positive slopes don't guarantee global monotonicity.
    
    Args:
        control_coeffs: Control coefficient values (x-axis)
        output_values: Output values (y-axis) 
        alpha: Sensitivity parameter for exponential mapping (default=1.0)
    
    Returns:
        ndcg_score: NDCG score between 0 and 1 (1 = perfect global monotonicity)
        correlation: Traditional Spearman correlation for comparison
    """
    if len(control_coeffs) < 2 or np.std(output_values) < 1e-9:
        return np.nan, np.nan
    
    # Sort by control coefficients (x-values) to get actual sequence
    sorted_indices = np.argsort(control_coeffs)
    x_sorted = control_coeffs[sorted_indices]
    y_actual = output_values[sorted_indices]
    
    # Create ideal monotonic sequence by sorting y-values
    y_ideal = np.sort(y_actual)
    
    # Calculate global y-range for normalization
    y_range = np.max(y_actual) - np.min(y_actual)
    
    # Handle edge case: all y-values are identical (perfect "monotonicity" in a trivial sense)
    if y_range == 0:
        return 1.0, 1.0
    
    # Calculate normalized global deviations
    normalized_deviations = (y_actual - y_ideal) / y_range
    
    # Map deviations to relevance scores using exponential function
    # Positive deviation (actual > ideal) = higher score (exceeds expectation)
    # Negative deviation (actual < ideal) = lower score (below expectation)
    relevance_scores = np.exp(alpha * normalized_deviations)
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) is undefined, start from log2(2)
    
    # Calculate IDCG (Ideal DCG) - sort relevance scores in descending order
    ideal_relevance = np.sort(relevance_scores)[::-1]
    idcg = 0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)
    
    # Calculate NDCG
    if idcg == 0:
        ndcg_score = 0
    else:
        ndcg_score = dcg / idcg
    
    # Also calculate traditional Spearman correlation for comparison
    try:
        correlation, _ = spearmanr(control_coeffs, output_values)
    except:
        correlation = np.nan
    
    return ndcg_score, correlation

def compute_smoothness(output_values):
    if len(output_values) < 3: return np.nan, np.nan
    return np.mean(np.abs(np.diff(output_values))), np.mean(np.abs(np.diff(np.diff(output_values))))

def compute_control_efficacy(output_values):
    if len(output_values) < 1: return np.nan
    return np.max(output_values) - np.min(output_values)

def compute_control_intensity(control_coeffs):
    if len(control_coeffs) < 1: return np.nan
    return np.max(control_coeffs) - np.min(control_coeffs)

# --- NEW: Function to calculate metrics from RAW scatter data ---
def calculate_raw_metrics(detailed_results, choice_labels):
    """Calculates metrics on the disaggregated (raw) data points."""
    likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}
    
    raw_coeffs = []
    raw_likert_scores = []

    for result in detailed_results:
        raw_coeffs.append(result['control_level'])
        
        # Calculate Likert score for this individual data point
        choice_probs = result['choice_probabilities']
        total_prob = sum(choice_probs.values())
        if total_prob > 0:
            rescaled_probs = {label: p / total_prob for label, p in choice_probs.items()}
            score = sum(rescaled_probs.get(label, 0) * likert_weights.get(label, 0) for label in choice_labels)
            raw_likert_scores.append(score)
        else:
            raw_likert_scores.append(np.nan) # Append nan if probs are zero

    raw_coeffs = np.array(raw_coeffs)
    raw_likert_scores = np.array(raw_likert_scores)
    
    # Filter out any nans before metric calculation
    valid_mask = ~np.isnan(raw_likert_scores)
    
    # Calculate metrics on this raw data
    ndcg_score, correlation = compute_monotonicity(raw_coeffs[valid_mask], raw_likert_scores[valid_mask])
    
    # Smoothness doesn't make sense for unordered scatter data, so we report it as N/A
    # Efficacy and Intensity are properties of the overall range, so we use the averaged data for them.
    return {
        "ndcg": ndcg_score,
        "rho": correlation,
        "smoothness_1": np.nan,
        "smoothness_2": np.nan,
    }


# --- Plotting Functions (Updated and New) ---

def _create_metrics_textbox(ax, metrics, is_raw=False):
    """Helper to create a formatted text box with metrics on a plot."""
    title = r"$\bf{Raw\ Data\ Metrics}$" if is_raw else r"$\bf{Averaged\ Metrics}$"
    
    text_lines = [title]
    if not np.isnan(metrics.get('ndcg', np.nan)):
        text_lines.append(f"NDCG Monotonicity: {metrics['ndcg']:.3f}")
    if not np.isnan(metrics.get('rho', np.nan)):
        text_lines.append(f"Spearman ρ: {metrics['rho']:.3f}")

    # Only show these metrics for averaged data
    if not is_raw:
        text_lines.extend([
            f"Smoothness (Δ¹): {metrics['smoothness_1']:.4f}",
            f"Smoothness (Δ²): {metrics['smoothness_2']:.4f}",
            f"Efficacy (Output Δ): {metrics['efficacy']:.3f}",
            f"Intensity (Input Δ): {metrics['intensity']:.2f}",
        ])
    
    props = dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.9)
    ax.text(0.03, 0.97, "\n".join(text_lines), transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, zorder=10)

def setup_dual_axis_plot(ax, control_levels, title_suffix, plot_title):
    # ... (This function remains unchanged)
    ax.set_title(f"{plot_title}\n{title_suffix}", fontsize=15)
    ax.set_xlabel('Control Coefficient (Rescaled)', fontsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    min_coeff, max_coeff = np.min(control_levels), np.max(control_levels)
    if min_coeff <= 0 <= max_coeff and max_coeff > min_coeff:
        zero_pos_rescaled = (0 - min_coeff) / (max_coeff - min_coeff)
        ax.axvline(x=zero_pos_rescaled, color='black', linestyle='--', linewidth=1.2, label='No Control')
    ax2 = ax.twiny()
    ax2.set_xlabel('Control Coefficient (Actual Value)', fontsize=12)
    ax.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_xticks(ax.get_xticks())
    new_tick_labels = [f"{min_coeff + tick * (max_coeff - min_coeff):.2f}" for tick in ax.get_xticks()]
    ax2.set_xticklabels(new_tick_labels)
    if min_coeff <= 0 <= max_coeff: ax.legend(loc='lower right')

# --- Averaged Data Plots ---
def plot_likert_with_metrics(ax, rescaled_levels, likert_scores, metrics, control_levels, title_suffix):
    # ... (This function remains unchanged)
    setup_dual_axis_plot(ax, control_levels, title_suffix, "Likert Score Analysis (Averaged)")
    ax.plot(rescaled_levels, likert_scores, marker='o', linestyle='-', color='darkgreen', zorder=5)
    ax.set_ylabel('Average Likert Score (4=A, 0=E)', fontsize=12)
    _create_metrics_textbox(ax, metrics)
    
# --- NEW Scatter Data Plots ---
def plot_likert_scatter_with_metrics(ax, detailed_results, raw_metrics, control_levels, title_suffix, choice_labels):
    """Plots the Likert score scatter vs. rescaled control, annotated with RAW metrics."""
    setup_dual_axis_plot(ax, control_levels, title_suffix, "Likert Score Analysis (Scatter)")
    
    min_coeff, max_coeff = np.min(control_levels), np.max(control_levels)
    coef_range = max_coeff - min_coeff if max_coeff > min_coeff else 1.0

    # Extract scatter data
    raw_coeffs = [res['control_level'] for res in detailed_results]
    rescaled_coeffs = [(c - min_coeff) / coef_range for c in raw_coeffs]
    
    likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}
    raw_scores = []
    for res in detailed_results:
        total_prob = sum(res['choice_probabilities'].values())
        score = sum(p / total_prob * likert_weights.get(l, 0) for l, p in res['choice_probabilities'].items()) if total_prob > 0 else np.nan
        raw_scores.append(score)

    ax.scatter(rescaled_coeffs, raw_scores, alpha=0.2, color='darkorange', zorder=5)
    ax.set_ylabel('Individual Likert Score (4=A, 0=E)', fontsize=12)
    
    # Add the raw metrics text box
    # We pass efficacy/intensity from the avg metrics, as they are range properties
    _create_metrics_textbox(ax, raw_metrics, is_raw=True)


# --- Main Evaluation Script ---
def evaluate_results(results_dir, plot=True):
    model_name = os.path.basename(results_dir)
    print(f"--- Evaluating results for model: {model_name} ---")

    # Find all data files
    avg_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_plot_data.json')])
    scatter_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_scatter_data.json')])

    # Create a map for easy lookup
    scatter_map = {f.replace('_scatter_data.json', ''): f for f in scatter_files}

    for fname in avg_files:
        basename = fname.replace('_plot_data.json', '')
        
        # --- Load Averaged Data ---
        with open(os.path.join(results_dir, fname), 'r') as f: data = json.load(f)
        control_levels = np.array(data['control_levels_actual'])
        rescaled_levels = np.array(data['control_levels_rescaled'])
        likert_scores = np.array(data['likert_scores'])
        
        title_suffix = basename.replace(f"{model_name}_", "")
        print(f"\n=== Evaluating: {title_suffix} ===")
        
        # --- 1. Calculate and Plot AVERAGED Metrics ---
        ndcg_score, correlation = compute_monotonicity(control_levels, likert_scores)
        avg_metrics = {
            "ndcg": ndcg_score,
            "rho": correlation,
            "smoothness_1": compute_smoothness(likert_scores)[0],
            "smoothness_2": compute_smoothness(likert_scores)[1],
            "efficacy": compute_control_efficacy(likert_scores),
            "intensity": compute_control_intensity(control_levels),
        }
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        plot_likert_with_metrics(ax1, rescaled_levels, likert_scores, avg_metrics, control_levels, title_suffix)
        fig1.tight_layout(); fig1.savefig(os.path.join(results_dir, fname.replace('_plot_data.json', '_likert_summary.png')))
        plt.close(fig1)

        # --- 2. Calculate and Plot RAW/SCATTER Metrics ---
        if basename in scatter_map:
            scatter_fname = scatter_map[basename]
            with open(os.path.join(results_dir, scatter_fname), 'r') as f: s_data = json.load(f)
            
            detailed_results = s_data['detailed_results']
            choice_labels = s_data['choice_labels']

            raw_metrics = calculate_raw_metrics(detailed_results, choice_labels)
            # Inherit range properties from avg_metrics as they apply to the whole experiment
            raw_metrics['efficacy'] = avg_metrics['efficacy']
            raw_metrics['intensity'] = avg_metrics['intensity']

            fig2, ax2 = plt.subplots(figsize=(12, 8))
            plot_likert_scatter_with_metrics(ax2, detailed_results, raw_metrics, control_levels, title_suffix, choice_labels)
            fig2.tight_layout(); fig2.savefig(os.path.join(results_dir, scatter_fname.replace('_scatter_data.json', '_likert_scatter_summary.png')))
            plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate control experiment results and generate summary plots.")
    parser.add_argument("results_dir", type=str, help="The path to the directory containing the '_plot_data.json' files for a specific model.")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found at '{args.results_dir}'")
    else:
        evaluate_results(args.results_dir)