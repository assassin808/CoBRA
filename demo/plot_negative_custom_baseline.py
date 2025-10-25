#!/usr/bin/env python3
"""
Baseline negative-only sentiment plotting script.

Replicates the structure of plot_negative_custom.py but operates on the
baseline CSV (nl_level bias levels) instead of coefficient / CBI mapping.

Produces:
 1. Sentiment vs Number of Negative Posts (lines = bias levels)
 2. Sentiment vs Bias Level (x ordinal 0-4, lines = selected group sizes)
 3. Combined 1x2: (subset bias levels on left, full bias-level plot on right)

Bias level mapping (ordered):
  level0_resist          -> No bias
  level1_somewhat_enhance -> Little bias
  level2_neutral          -> Original (dashed gray line)
  level3_strong_enhance   -> Some bias
  level4_max_enhance      -> Much bias

Styling:
 - Comic Sans MS if available (fallback DejaVu Sans)
 - Linewidth=3, markersize=7.5, alpha=0.65 (to match plot_negative_custom)
 - Neutral 'Original' line: dashed ('--'), color gray (#555555), alpha=0.65
 - Shading optional: none | std | ci95 (default ci95)

Outputs (PNG + PDF + EPS for first figure only, PNG for others):
  baseline_bias_vs_group_size.*
  baseline_sentiment_vs_biaslevel.png
  baseline_combined_subset_full.png
"""

from __future__ import annotations
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

BIAS_ORDER = [
    "level0_resist",
    "level1_somewhat_enhance",
    "level2_neutral",
    "level3_strong_enhance",
    "level4_max_enhance",
]
DISPLAY_MAP = {
    "level0_resist": "No bias",
    "level1_somewhat_enhance": "Little bias",
    "level2_neutral": "Original",
    "level3_strong_enhance": "Some bias",
    "level4_max_enhance": "Much bias",
}
# Colors for non-neutral levels (in order excluding neutral). Neutral gets gray dashed.
COLOR_MAP = {
    "level0_resist": "#A2D94D",
    "level1_somewhat_enhance": "#22BDD2",
    "level3_strong_enhance": "#9368AB",
    "level4_max_enhance": "#FF6B6B",
}
NEUTRAL_COLOR = "#555555"


def ensure_comic_sans(ttf_path: str = "") -> bool:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "Comic Sans MS" in available_fonts:
        return True
    if ttf_path and os.path.isfile(ttf_path):
        try:
            font_manager.fontManager.addfont(ttf_path)
            font_manager._rebuild()
            if "Comic Sans MS" in {f.name for f in font_manager.fontManager.ttflist}:
                print(f"[Font] Loaded Comic Sans MS from {ttf_path}")
                return True
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"[Font] Failed to load provided path: {e}")
    common_paths = [
        '/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/comic.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/comicbd.ttf',
        '/System/Library/Fonts/Comic Sans MS.ttf',
        '/Windows/Fonts/comic.ttf',
        os.path.expanduser('~/.local/share/fonts/Comic_Sans_MS.ttf'),
        os.path.expanduser('~/.fonts/Comic_Sans_MS.ttf'),
        os.path.expanduser('~/Library/Fonts/Comic Sans MS.ttf'),
    ]
    for path in common_paths:
        if os.path.isfile(path):
            try:
                font_manager.fontManager.addfont(path)
                font_manager._rebuild()
                if "Comic Sans MS" in {f.name for f in font_manager.fontManager.ttflist}:
                    print(f"[Font] Loaded Comic Sans MS from {path}")
                    return True
            except Exception as e:  # pragma: no cover
                print(f"[Font] Failed from {path}: {e}")
    return False


def setup_font(ttf_path: str = ""):
    if ensure_comic_sans(ttf_path):
        fam = "Comic Sans MS"
        print("[Font] Using Comic Sans MS")
    else:
        fam = "DejaVu Sans"
        print("[Font] Comic Sans MS not found. Using DejaVu Sans.")
    mpl.rcParams['font.family'] = fam
    mpl.rcParams['font.size'] = 17
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'


def _compute_band(mean_vals, std_vals, n_vals, shading_mode):
    if shading_mode == 'none' or std_vals is None:
        return None, None
    if shading_mode == 'ci95':
        if n_vals is not None and np.all(n_vals > 0):
            sem = std_vals / np.sqrt(n_vals)
            band = 1.96 * sem
        else:
            band = std_vals
    elif shading_mode == 'std':
        band = std_vals
    else:
        return None, None
    upper = np.clip(mean_vals + band, -1, 1)
    lower = np.clip(mean_vals - band, -1, 1)
    return lower, upper


def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {'pool', 'group_size', 'nl_level', 'mean_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    neg = df[df['pool'] == 'negative'].copy()
    if neg.empty:
        raise ValueError("No negative pool rows found")
    # Keep only recognized bias levels
    neg = neg[neg['nl_level'].isin(BIAS_ORDER)]
    return neg


def plot_bias_vs_group_size(neg_df: pd.DataFrame, output_dir: str, shading_mode: str):
    print("[Plot] Figure 1: Sentiment vs Number of Negative Posts (bias levels)")
    fig, ax = plt.subplots(figsize=(9.6, 7.2))

    for level in BIAS_ORDER:
        sub = neg_df[neg_df['nl_level'] == level].sort_values('group_size')
        if sub.empty:
            continue
        mean_vals = sub['mean_score'].values
        std_vals = sub['std_score'].values if 'std_score' in sub.columns else None
        n_vals = sub['n'].values if 'n' in sub.columns else None
        x = sub['group_size'].values
        label = DISPLAY_MAP[level]
        if level == 'level2_neutral':
            ax.plot(
                x, mean_vals,
                linestyle='--', color=NEUTRAL_COLOR, marker='o',
                linewidth=3, markersize=7.5, alpha=0.65, label=label
            )
        else:
            ax.plot(
                x, mean_vals,
                marker='o', color=COLOR_MAP[level], label=label,
                linewidth=3, markersize=7.5, alpha=0.65
            )
        lower, upper = _compute_band(mean_vals, std_vals, n_vals, shading_mode)
        if lower is not None:
            shade_color = NEUTRAL_COLOR if level == 'level2_neutral' else COLOR_MAP[level]
            ax.fill_between(x, lower, upper, color=shade_color, alpha=0.12, linewidth=0)

    ax.set_xlabel('Number of Negative Posts', fontsize=17, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax.minorticks_on()

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(12)
        t.set_fontweight('normal')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        header = Line2D([], [], linestyle='None', marker='', label='Bias Levels:')
        handles = [header] + handles
        labels = ['Bias Levels:'] + labels
    leg = ax.legend(handles, labels, loc='best', frameon=True, fontsize=17)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out_png = os.path.join(output_dir, 'baseline_bias_vs_group_size.png')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.eps'), dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {out_png} (+ pdf, eps)")
    plt.close(fig)


def plot_sentiment_vs_biaslevel(neg_df: pd.DataFrame, output_dir: str, shading_mode: str):
    print("[Plot] Figure 2: Sentiment vs Bias Level (lines = group sizes)")
    fig, ax = plt.subplots(figsize=(9.6, 7.2))

    target_group_sizes = [1, 3, 5, 7, 9, 11]
    available = sorted(neg_df['group_size'].unique())
    group_sizes = [g for g in target_group_sizes if g in available]
    if not group_sizes:
        group_sizes = available
    print(f"[Plot] Using group sizes: {group_sizes}")

    # Precompute x positions 0..4
    x_positions = np.arange(len(BIAS_ORDER))
    tick_labels = [DISPLAY_MAP[l] for l in BIAS_ORDER]
    cmap = mpl.cm.get_cmap('viridis')

    for idx, gsz in enumerate(group_sizes):
        sub = neg_df[neg_df['group_size'] == gsz]
        if sub.empty:
            continue
        means = []
        stds = []
        ns = []
        colors_line = []  # Not used per point, but keep for potential extension
        for level in BIAS_ORDER:
            row = sub[sub['nl_level'] == level]
            if row.empty:
                means.append(np.nan)
                stds.append(np.nan)
                ns.append(np.nan)
                continue
            means.append(row['mean_score'].values[0])
            stds.append(row['std_score'].values[0] if 'std_score' in row.columns else np.nan)
            ns.append(row['n'].values[0] if 'n' in row.columns else np.nan)
        means = np.array(means)
        stds = np.array(stds)
        ns = np.array(ns)
        color = cmap(idx / max(len(group_sizes) - 1, 1))
        ax.plot(
            x_positions,
            means,
            marker='o',
            label=f'n={gsz}',
            color=color,
            linewidth=3,
            markersize=7.5,
            alpha=0.65,
        )
        # We do not shade across bias levels for group-size lines to keep uncluttered.

    ax.set_xlabel('Bias Level', fontsize=17, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=15)
    ax.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax.minorticks_on()

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(12)
        t.set_fontweight('normal')
    leg = ax.legend(loc='best', frameon=True, fontsize=17)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(output_dir, 'baseline_sentiment_vs_biaslevel.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {out}")
    plt.close(fig)


def plot_combined(neg_df: pd.DataFrame, output_dir: str, shading_mode: str):
    print("[Plot] Combined figure (subset + full)")
    # Subset choose: No bias, Original, Much bias (levels 0,2,4)
    subset_levels = ["level0_resist", "level2_neutral", "level4_max_enhance"]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.6, 4.8))

    # Left panel
    for level in subset_levels:
        sub = neg_df[neg_df['nl_level'] == level].sort_values('group_size')
        if sub.empty:
            continue
        mean_vals = sub['mean_score'].values
        std_vals = sub['std_score'].values if 'std_score' in sub.columns else None
        n_vals = sub['n'].values if 'n' in sub.columns else None
        x = sub['group_size'].values
        label = DISPLAY_MAP[level]
        if level == 'level2_neutral':
            ax_left.plot(x, mean_vals, linestyle='--', color=NEUTRAL_COLOR, marker='o',
                         linewidth=3, markersize=7.5, alpha=0.65, label=label)
        else:
            ax_left.plot(x, mean_vals, marker='o', color=COLOR_MAP.get(level, '#666666'),
                         linewidth=3, markersize=7.5, alpha=0.65, label=label)
        lower, upper = _compute_band(mean_vals, std_vals, n_vals, shading_mode)
        if lower is not None:
            shade_color = NEUTRAL_COLOR if level == 'level2_neutral' else COLOR_MAP.get(level, '#666666')
            ax_left.fill_between(x, lower, upper, color=shade_color, alpha=0.12, linewidth=0)
    ax_left.set_xlabel('Number of Negative Posts', fontsize=17, fontweight='bold')
    ax_left.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax_left.set_ylim(-1, 1)
    ax_left.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax_left.minorticks_on()
    for t in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        t.set_fontsize(12)
        t.set_fontweight('normal')
    leg_left = ax_left.legend(loc='best', frameon=True, fontsize=17)
    for txt in leg_left.get_texts():
        txt.set_fontweight('bold')

    # Right panel reuse Figure 2 logic simplified: Sentiment vs Bias Level lines group sizes
    target_group_sizes = [1, 3, 5, 7, 9, 11]
    available = sorted(neg_df['group_size'].unique())
    group_sizes = [g for g in target_group_sizes if g in available]
    if not group_sizes:
        group_sizes = available
    x_positions = np.arange(len(BIAS_ORDER))
    tick_labels = [DISPLAY_MAP[l] for l in BIAS_ORDER]
    cmap = mpl.cm.get_cmap('viridis')
    for idx, gsz in enumerate(group_sizes):
        sub = neg_df[neg_df['group_size'] == gsz]
        if sub.empty:
            continue
        means = []
        for level in BIAS_ORDER:
            row = sub[sub['nl_level'] == level]
            means.append(row['mean_score'].values[0] if not row.empty else np.nan)
        means = np.array(means)
        color = cmap(idx / max(len(group_sizes) - 1, 1))
        ax_right.plot(
            x_positions,
            means,
            marker='o',
            label=f'n={gsz}',
            color=color,
            linewidth=3,
            markersize=7.5,
            alpha=0.65,
        )
    ax_right.set_xlabel('Bias Level', fontsize=17, fontweight='bold')
    ax_right.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax_right.set_ylim(-1, 1)
    ax_right.set_xticks(x_positions)
    ax_right.set_xticklabels(tick_labels, rotation=15)
    ax_right.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax_right.minorticks_on()
    for t in ax_right.get_xticklabels() + ax_right.get_yticklabels():
        t.set_fontsize(12)
        t.set_fontweight('normal')
    leg_right = ax_right.legend(loc='best', frameon=True, fontsize=17)
    for txt in leg_right.get_texts():
        txt.set_fontweight('bold')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(output_dir, 'baseline_combined_subset_full.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"[Plot] Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline negative-only bias plots")
    parser.add_argument('--summary-csv', required=True, help='Path to baseline sentiment summary CSV')
    parser.add_argument('--output-dir', default='plots_baseline', help='Output directory')
    parser.add_argument('--comic-sans-ttf', default='', help='Path to Comic Sans MS TTF file')
    parser.add_argument('--shading', choices=['none', 'std', 'ci95'], default='ci95',
                        help='Uncertainty shading mode (default: ci95)')
    args = parser.parse_args()

    if not os.path.exists(args.summary_csv):
        print(f"[Error] CSV not found: {args.summary_csv}")
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        neg_df = load_and_validate(args.summary_csv)
    except Exception as e:
        print(f"[Error] {e}")
        return 1

    print(f"[Data] Rows (negative): {len(neg_df)}")
    print(f"[Data] Group sizes: {sorted(neg_df['group_size'].unique())}")

    setup_font(args.comic_sans_ttf)

    plot_bias_vs_group_size(neg_df, args.output_dir, args.shading)
    plot_sentiment_vs_biaslevel(neg_df, args.output_dir, args.shading)
    plot_combined(neg_df, args.output_dir, args.shading)

    print(f"[Complete] Baseline plots saved to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
