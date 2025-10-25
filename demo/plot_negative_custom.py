#!/usr/bin/env python3
"""
Standalone script for generating custom negative-only CBI plots.

This script reads sentiment summary data and generates two custom plots:
1. Sentiment vs Number of Negative Posts (group_size) with curves for CBI values
2. Sentiment vs CBI with curves for different group_size values

Features:
- Comic Sans MS font support with automatic fallback
- Grid lines and custom styling
- Configurable coefficient-to-CBI mapping
- High-resolution output (200 DPI)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


def ensure_comic_sans(ttf_path: str = "") -> bool:
    """Try to load Comic Sans MS font from various sources."""
    # First check if already available
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if 'Comic Sans MS' in available_fonts:
        return True
    
    # Try provided TTF path
    if ttf_path and os.path.isfile(ttf_path):
        try:
            font_manager.fontManager.addfont(ttf_path)
            font_manager._rebuild()
            available_fonts = {f.name for f in font_manager.fontManager.ttflist}
            if 'Comic Sans MS' in available_fonts:
                print(f"[Font] Successfully loaded Comic Sans from: {ttf_path}")
                return True
        except Exception as e:
            print(f"[Font] Failed to load provided TTF '{ttf_path}': {e}")
    
    # Try common system locations
    common_paths = [
        '/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/comic.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/comicbd.ttf',
        '/System/Library/Fonts/Comic Sans MS.ttf',  # macOS
        '/Windows/Fonts/comic.ttf',  # Windows
        os.path.expanduser('~/.local/share/fonts/Comic_Sans_MS.ttf'),
        os.path.expanduser('~/.fonts/Comic_Sans_MS.ttf'),
        os.path.expanduser('~/Library/Fonts/Comic Sans MS.ttf'),  # macOS user
    ]
    
    for path in common_paths:
        if os.path.isfile(path):
            try:
                font_manager.fontManager.addfont(path)
                font_manager._rebuild()
                available_fonts = {f.name for f in font_manager.fontManager.ttflist}
                if 'Comic Sans MS' in available_fonts:
                    print(f"[Font] Successfully loaded Comic Sans from: {path}")
                    return True
            except Exception as e:
                print(f"[Font] Failed to load from {path}: {e}")
    
    return False


def setup_font(comic_sans_ttf: str = ""):
    """Configure matplotlib font settings with Comic Sans MS or fallback."""
    have_comic_sans = ensure_comic_sans(comic_sans_ttf)
    
    if have_comic_sans:
        font_family = 'Comic Sans MS'
        print("[Font] Using Comic Sans MS")
    else:
        font_family = 'DejaVu Sans'
        print("[Font] Comic Sans MS not available. Using DejaVu Sans fallback.")
        print("[Font] To install Comic Sans MS:")
        print("  Ubuntu/Debian: sudo apt-get install ttf-mscorefonts-installer")
        print("  Or provide path with --comic-sans-ttf /path/to/comic.ttf")
    
    # Set matplotlib font parameters
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['font.size'] = 17
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'


def _compute_band(mean_vals, std_vals, n_vals, shading_mode):
    """Return (lower, upper) arrays for shading or (None, None) if disabled.
    shading_mode: 'none' | 'std' | 'ci95'
    mean/std/n arrays assumed aligned. Clipped to [-1,1]."""
    if shading_mode == 'none' or std_vals is None:
        return None, None
    if shading_mode == 'ci95':
        if n_vals is not None and np.all(n_vals > 0):
            sem = std_vals / np.sqrt(n_vals)
            band = 1.96 * sem
        else:
            # Fallback to std if n not available
            band = std_vals
    elif shading_mode == 'std':
        band = std_vals
    else:
        return None, None
    upper = np.clip(mean_vals + band, -1, 1)
    lower = np.clip(mean_vals - band, -1, 1)
    return lower, upper


def _fmt_coeff(c: float) -> str:
    """Format coefficient with + sign if positive, plain 0.0 if zero, minus preserved."""
    if c > 0:
        return f"+{c:.1f}"
    if abs(c) < 1e-9:
        return f"{0:.1f}"
    return f"{c:.1f}"


def plot_negative_custom(summary_df, coeff_map, coeff_order, output_dir, shading_mode='ci95'):
    """Generate custom negative-only plots with specified styling.
    shading_mode controls uncertainty band: 'none', 'std', or 'ci95'."""
    
    # Filter for negative pool only
    neg_df = summary_df[summary_df['pool'] == 'negative'].copy()
    
    # Keep only specified coefficients that exist in data
    available_coeffs = set(neg_df['coeff'].unique())
    filtered_order = [c for c in coeff_order if c in available_coeffs]
    
    if not filtered_order:
        print(f"[Error] None of the requested coefficients {coeff_order} found in data.")
        print(f"[Error] Available coefficients: {sorted(available_coeffs)}")
        return
    
    neg_df = neg_df[neg_df['coeff'].isin(filtered_order)]
    
    if neg_df.empty:
        print("[Error] No negative data found after filtering.")
        return
    
    # Map coefficients to CBI values
    neg_df['CBI'] = neg_df['coeff'].map(coeff_map)
    neg_df = neg_df.dropna(subset=['CBI'])  # Remove unmapped coefficients
    
    if neg_df.empty:
        print("[Error] No data remaining after CBI mapping.")
        return
    
    # Color palette as specified (extended for 6 coefficients)
    palette = ["#A2D94D", "#22BDD2", "#859ED7", "#9368AB", "#F8A87B", "#FF6B6B"]
    
    print(f"[Plot] Generating plots for {len(filtered_order)} coefficients: {filtered_order}")
    
    # Figure 1: Sentiment vs Number of Negative Posts
    print("[Plot] Creating Figure 1: Sentiment vs Number of Negative Posts")
    fig1, ax1 = plt.subplots(figsize=(9.6, 7.2))
    
    for i, coeff in enumerate(filtered_order):
        cbi = coeff_map.get(coeff)
        if cbi is None:
            continue

        subset = neg_df[neg_df['coeff'] == coeff].sort_values('group_size')
        if subset.empty:
            continue

        color = palette[i % len(palette)]
        mean_vals = subset['mean_score'].values
        std_vals = subset['std_score'].values if 'std_score' in subset.columns else None
        x_vals = subset['group_size'].values
        # Line
        ax1.plot(
            x_vals,
            mean_vals,
            marker='o',
            label=f'{cbi:.2f} ({_fmt_coeff(coeff)})',
            color=color,
            linewidth=3,
            markersize=7.5,
            alpha=0.65,
        )
        # Shading (optional)
        n_vals = subset['n'].values if 'n' in subset.columns else None
        lower, upper = _compute_band(mean_vals, std_vals, n_vals, shading_mode)
        if lower is not None:
            ax1.fill_between(x_vals, lower, upper, color=color, alpha=0.12, linewidth=0)
    
    # Styling
    ax1.set_xlabel('Number of Negative Posts', fontsize=17, fontweight='bold')
    ax1.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax1.set_ylim(-1, 1)  # Set y-axis range from -1 to 1
    # Simplified, lighter grid (no minor grid lines to reduce density)
    ax1.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax1.minorticks_on()  # Keep minor ticks for readability (but no minor grid)
    
    # Tick formatting
    for tick in ax1.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('normal')
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('normal')
    
    # Legend: separate header entry (no line) before colored lines
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        header_handle = Line2D([], [], linestyle='None', marker='', label='CBI (coef):')
        handles = [header_handle] + handles
        labels = ['CBI (coef.)'] + labels
        legend1 = ax1.legend(
            handles,
            labels,
            loc='best',
            frameon=True,
            fontsize=17,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.2,
            columnspacing=0.8,
        )
        for txt in legend1.get_texts():
            txt.set_fontweight('bold')
    
    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save Figure 1
    output1 = os.path.join(output_dir, 'negative_cbi_vs_group_size.png')
    fig1.savefig(output1, dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {output1}")
    #  save for pdf and eps
    output1_pdf = os.path.join(output_dir, 'negative_cbi_vs_group_size.pdf')
    fig1.savefig(output1_pdf, dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {output1_pdf}")

    output1_eps = os.path.join(output_dir, 'negative_cbi_vs_group_size.eps')
    fig1.savefig(output1_eps, dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {output1_eps}")

    # Figure 2: Sentiment vs CBI (only specific group sizes: 1,3,5,7,9,11,13,15)
    print("[Plot] Creating Figure 2: Sentiment vs CBI")
    fig2, ax2 = plt.subplots(figsize=(9.6, 7.2))
    
    # Filter to only specific group sizes as requested
    target_group_sizes = [1, 3, 5, 7, 9, 11]
    available_group_sizes = sorted(neg_df['group_size'].unique())
    group_sizes = [g for g in target_group_sizes if g in available_group_sizes]
    
    if not group_sizes:
        print(f"[Warning] None of target group sizes {target_group_sizes} found in data.")
        print(f"[Warning] Available group sizes: {available_group_sizes}")
        group_sizes = available_group_sizes  # Fallback to all available
    
    print(f"[Plot] Using group sizes: {group_sizes}")
    cmap = mpl.cm.get_cmap('viridis')
    
    for idx, group_size in enumerate(group_sizes):
        subset = neg_df[neg_df['group_size'] == group_size].copy()
        
        # Reorder by coefficient order
        subset = subset.set_index('coeff').reindex(filtered_order).reset_index()
        subset = subset.dropna(subset=['mean_score'])  # Remove missing data points
        
        if subset.empty:
            continue
        
        # Get CBI values for x-axis
        cbi_values = [coeff_map[c] for c in filtered_order if c in subset['coeff'].values]
        scores = subset['mean_score'].values[:len(cbi_values)]
        
        color = cmap(idx / max(len(group_sizes) - 1, 1))
        ax2.plot(
            cbi_values,
            scores,
            marker='o',
            label=f'n={group_size}',
            color=color,
            linewidth=3,
            markersize=7.5,
            alpha=0.65,
        )
    
    # Styling
    ax2.set_xlabel('CBI', fontsize=17, fontweight='bold')
    ax2.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax2.set_ylim(-1, 1)  # Set y-axis range from -1 to 1
    # Simplified, lighter grid (no minor grid lines to reduce density)
    ax2.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax2.minorticks_on()  # Keep minor ticks for readability
    
    # Tick formatting
    for tick in ax2.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('normal')
    for tick in ax2.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('normal')
    
    # Legend
    legend2 = ax2.legend(loc='best', frameon=True, fontsize=17)
    for text in legend2.get_texts():
        text.set_fontweight('bold')
    
    fig2.tight_layout(rect=[0, 0.07, 1, 1])
    
    # Save Figure 2
    output2 = os.path.join(output_dir, 'negative_cbi_vs_coeff.png')
    fig2.savefig(output2, dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {output2}")
    
    plt.close(fig1)
    plt.close(fig2)

    # -------------------------------------------------------------
    # Combined 1x2 Plot
    # Left panel: Figure 1 but ONLY subset of first four coefficients
    # Right panel: Figure 2 (Sentiment vs CBI) using full data
    # -------------------------------------------------------------
    try:
        subset_coeffs = [-0.625, -0.247, 0, 0.736]
        subset_cbis = [coeff_map.get(c) for c in subset_coeffs if c in coeff_map]
        # Filter data for subset panel
        subset_df = neg_df[neg_df['coeff'].isin(subset_coeffs)].copy()
        if not subset_df.empty:
            fig_comb, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
            ax_left, ax_right = axes

            # Left panel: group_size vs mean_score for subset coefficients
            for i, coeff in enumerate(subset_coeffs):
                if coeff not in subset_df['coeff'].values:
                    continue
                cbi_val = coeff_map.get(coeff)
                sub_sub = subset_df[subset_df['coeff'] == coeff].sort_values('group_size')
                if sub_sub.empty:
                    continue
                color = palette[i % len(palette)]
                mean_vals = sub_sub['mean_score'].values
                std_vals = sub_sub['std_score'].values if 'std_score' in sub_sub.columns else None
                x_vals = sub_sub['group_size'].values
                ax_left.plot(
                    x_vals,
                    mean_vals,
                    marker='o',
                    label=f'{cbi_val:.2f} ({_fmt_coeff(coeff)})',
                    color=color,
                    linewidth=3,
                    markersize=7.5,
                    alpha=0.65,
                )
                n_vals = sub_sub['n'].values if 'n' in sub_sub.columns else None
                lower, upper = _compute_band(mean_vals, std_vals, n_vals, shading_mode)
                if lower is not None:
                    ax_left.fill_between(x_vals, lower, upper, color=color, alpha=0.12, linewidth=0)
            ax_left.set_xlabel('Number of Negative Posts', fontsize=17, fontweight='bold')
            ax_left.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
            ax_left.set_ylim(-1, 1)
            ax_left.grid(True, which='major', alpha=0.25, linewidth=0.8)
            ax_left.minorticks_on()
            for tick in ax_left.get_xticklabels() + ax_left.get_yticklabels():
                tick.set_fontsize(12)
                tick.set_fontweight('normal')
            # Legend for combined left panel with separate header
            handles_left, labels_left = ax_left.get_legend_handles_labels()
            if handles_left:
                header_handle_left = Line2D([], [], linestyle='None', marker='', label='CBI (coef):')
                handles_left = [header_handle_left] + handles_left
                labels_left = ['CBI (coef):'] + labels_left
                leg_left = ax_left.legend(
                    handles_left,
                    labels_left,
                    loc='best',
                    frameon=True,
                    fontsize=17,
                    handlelength=1.2,
                    handletextpad=0.5,
                    borderpad=0.2,
                    columnspacing=0.8,
                )
                for t in leg_left.get_texts():
                    t.set_fontweight('bold')

            # Right panel: full Sentiment vs CBI for selected group sizes
            target_group_sizes = [1, 3, 5, 7, 9, 11]
            available_group_sizes = sorted(neg_df['group_size'].unique())
            group_sizes_full = [g for g in target_group_sizes if g in available_group_sizes]
            if not group_sizes_full:
                group_sizes_full = available_group_sizes
            cmap = mpl.cm.get_cmap('viridis')
            for idx, gsz in enumerate(group_sizes_full):
                sub = neg_df[neg_df['group_size'] == gsz].copy()
                if sub.empty:
                    continue
                sub = sub.set_index('coeff').reindex(coeff_order).reset_index()
                sub = sub.dropna(subset=['mean_score'])
                if sub.empty:
                    continue
                # Build aligned sequences
                cbi_vals_full = []
                scores_full = []
                stds_full = []
                ns_full = []
                for c in coeff_order:
                    row = sub[sub['coeff'] == c]
                    if row.empty:
                        continue
                    cbi_val = coeff_map.get(c)
                    if cbi_val is None:
                        continue
                    cbi_vals_full.append(cbi_val)
                    scores_full.append(row['mean_score'].values[0])
                    stds_full.append(row['std_score'].values[0] if 'std_score' in row.columns else np.nan)
                    ns_full.append(row['n'].values[0] if 'n' in row.columns else np.nan)
                if not cbi_vals_full:
                    continue
                cbi_vals_full = np.array(cbi_vals_full)
                scores_full = np.array(scores_full)
                stds_full = np.array(stds_full)
                ns_full = np.array(ns_full)
                color = cmap(idx / max(len(group_sizes_full) - 1, 1))
                ax_right.plot(
                    cbi_vals_full,
                    scores_full,
                    marker='o',
                    label=f'n={gsz}',
                    color=color,
                    linewidth=3,
                    markersize=7.5,
                    alpha=0.65,
                )
                if not np.all(np.isnan(stds_full)):
                    lower, upper = _compute_band(scores_full, stds_full, ns_full, shading_mode)
                    if lower is not None:
                        ax_right.fill_between(cbi_vals_full, lower, upper, color=color, alpha=0.12, linewidth=0)
            ax_right.set_xlabel('CBI', fontsize=17, fontweight='bold')
            ax_right.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
            ax_right.set_ylim(-1, 1)
            ax_right.grid(True, which='major', alpha=0.25, linewidth=0.8)
            ax_right.minorticks_on()
            for tick in ax_right.get_xticklabels() + ax_right.get_yticklabels():
                tick.set_fontsize(12)
                tick.set_fontweight('normal')
            leg_right = ax_right.legend(loc='best', frameon=True, fontsize=17)
            for t in leg_right.get_texts():
                t.set_fontweight('bold')

            fig_comb.tight_layout(rect=[0, 0.08, 1, 1])
            combined_path = os.path.join(output_dir, 'negative_combined_subset_full.png')
            fig_comb.savefig(combined_path, dpi=200, bbox_inches='tight')
            print(f"[Plot] Saved combined figure: {combined_path}")
            plt.close(fig_comb)
        else:
            print("[Plot] Skipping combined figure - subset coefficients missing in data.")
    except Exception as e:
        print(f"[Error] Failed to create combined figure: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate custom negative-only CBI plots")
    parser.add_argument('--summary-csv', required=True, 
                       help='Path to sentiment summary CSV file')
    parser.add_argument('--output-dir', default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--comic-sans-ttf', default='',
                       help='Path to Comic Sans MS TTF file')
    # parser.add_argument('--coeffs', 
    #                    default='-0.625,-0.247,0,0.736,1.2653',
    #                    help='Comma-separated coefficient list')
    # parser.add_argument('--cbi-values',
    #                    default='2.55,2.80,3.00,3.13,3.35',
    #                    help='Comma-separated CBI values corresponding to coefficients')
    parser.add_argument('--coeffs', 
                       default='-0.625,-0.247,0,0.43359375,0.736',
                       help='Comma-separated coefficient list')
    parser.add_argument('--cbi-values',
                       default='2.55,2.80,3.00,3.11,3.13',
                       help='Comma-separated CBI values corresponding to coefficients')
    parser.add_argument('--shading', choices=['none','std','ci95'], default='ci95',
                       help='Uncertainty shading mode: none | std | ci95 (default: ci95)')
    
    args = parser.parse_args()
    
    # Parse coefficient and CBI mappings
    try:
        coeffs = [float(x.strip()) for x in args.coeffs.split(',') if x.strip()]
        cbi_vals = [float(x.strip()) for x in args.cbi_values.split(',') if x.strip()]
        
        if len(coeffs) != len(cbi_vals):
            raise ValueError(f"Number of coefficients ({len(coeffs)}) must match CBI values ({len(cbi_vals)})")
        
        coeff_map = dict(zip(coeffs, cbi_vals))
        
    except ValueError as e:
        print(f"[Error] Invalid coefficient or CBI values: {e}")
        return 1
    
    # Check input file
    if not os.path.exists(args.summary_csv):
        print(f"[Error] Summary CSV not found: {args.summary_csv}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"[Data] Loading summary from: {args.summary_csv}")
    try:
        summary_df = pd.read_csv(args.summary_csv)
    except Exception as e:
        print(f"[Error] Failed to load CSV: {e}")
        return 1
    
    # Validate data structure
    required_cols = ['pool', 'group_size', 'coeff', 'mean_score']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        print(f"[Error] Missing required columns in CSV: {missing_cols}")
        print(f"[Error] Available columns: {list(summary_df.columns)}")
        return 1
    
    # Check for negative data
    neg_data = summary_df[summary_df['pool'] == 'negative']
    if neg_data.empty:
        print("[Error] No negative pool data found in summary CSV")
        return 1
    
    print(f"[Data] Found {len(neg_data)} negative data points")
    print(f"[Data] Available coefficients: {sorted(neg_data['coeff'].unique())}")
    print(f"[Data] Requested coefficients: {coeffs}")
    print(f"[Data] CBI mapping: {coeff_map}")
    
    # Setup font
    setup_font(args.comic_sans_ttf)
    
    # Generate plots
    plot_negative_custom(summary_df, coeff_map, coeffs, args.output_dir, shading_mode=args.shading)
    
    print(f"[Complete] Plots saved to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
