# Demo: Facebook Bandwagon Sentiment Experiment

**📖 Language / 语言**: [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

This folder contains a complete demonstration of RepE-based bias control using a realistic social media scenario.

## Overview

The demo simulates how bandwagon bias affects sentiment in social media post generation. Users see varying numbers of positive or negative posts in their "feed" and then generate a new post using a language model with controllable bandwagon bias.

## Files

### Core Experiment
- **`facebook_bandwagon_sentiment_experiment.py`** - Main experiment script
  - Trains RepE bandwagon direction from unified bias dataset
  - Dynamically discovers control coefficients 
  - Generates posts under different bias levels and feed sizes
  - Scores sentiment and produces analysis plots

### Data Generation  
- **`generate_facebook_posts.py`** - Creates synthetic Facebook-style posts
  - Uses OpenRouter API (GPT-4, Claude, Gemini, etc.) to generate diverse posts
  - Produces separate positive and negative post pools
  - Configurable quantities and selective regeneration

### Analysis & Plotting
- **`plot_negative_custom.py`** - Standalone plotting for CBI (Cognitive Bias Index) analysis
  - Generates publication-quality plots with Comic Sans MS styling
  - Two main plots: Sentiment vs Group Size, Sentiment vs CBI
  - Supports uncertainty bands (none/std/ci95)

- **`plot_negative_custom_baseline.py`** - Baseline bias level analysis
  - Similar plots but for discrete bias levels instead of coefficients
  - Useful for comparing with traditional prompt-based approaches

### Data Directory
- **`data/`** - Contains generated Facebook posts (created by `generate_facebook_posts.py`)
  - `facebook_positive_posts.json`
  - `facebook_negative_posts.json`

## Quick Start

### 1. Generate Data
First, create synthetic Facebook posts using OpenRouter:

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"

# Generate 100 positive + 100 negative posts (default)
python generate_facebook_posts.py

# Or customize quantities
python generate_facebook_posts.py --pos-total 200 --neg-total 500
```

### 2. Run Experiment
Run the main bandwagon sentiment experiment:

```bash
# Full experiment (requires GPU)
python facebook_bandwagon_sentiment_experiment.py

# Test mode (smaller scale)
python facebook_bandwagon_sentiment_experiment.py --test-mode

# Resume from existing generation data
python facebook_bandwagon_sentiment_experiment.py --reuse-generation

# Plot-only mode (reuse all existing data)
python facebook_bandwagon_sentiment_experiment.py --plot-only
```

### 3. Generate Analysis Plots
Create publication-quality plots:

```bash
# CBI analysis (requires experiment results)
python plot_negative_custom.py --summary-csv results/sentiment_summary.csv

# Baseline bias level analysis  
python plot_negative_custom_baseline.py --summary-csv results/baseline_sentiment_summary.csv
```

## Configuration

### Environment Variables (.env)
The experiment respects standard control module settings:
- `OPENROUTER_API_KEY` - For post generation
- `REASONING_BATCH_SIZE` - Batch size for generation
- `MAX_NEW_TOKENS` - Token limit for generated posts
- `TEMPERATURE` - Sampling temperature

### Command Line Options

**Main Experiment:**
- `--model` - HuggingFace model (default: mistral-7b)  
- `--test-mode` - Reduce scale for quick testing
- `--max-group-size N` - Maximum number of posts in feed
- `--feeds-per-size N` - Number of random feeds per size
- `--reuse-generation` / `--plot-only` - Skip expensive steps

**Post Generation:**
- `--total N` - Posts per category (positive/negative)
- `--pos-total` / `--neg-total` - Independent counts
- `--models` - OpenRouter models to use
- `--only-positive` / `--only-negative` - Selective regeneration

## Pipeline Details

### 1. RepE Training
- Uses existing unified bias dataset (bandwagon scenarios)
- Trains RepReader to identify bandwagon direction in model activations
- Evaluates accuracy across layers, selects top 15 for control

### 2. Coefficient Discovery  
- Dynamically finds effective control range using adaptive sampling
- Two-phase approach: boundary search + gradient-based refinement
- Ensures coefficients produce meaningful behavioral changes

### 3. Generation
- For each feed size (0-10) and coefficient:
  - Sample random feeds from positive/negative pools
  - Build prompt: "Based on these posts: [feed], what's your take?"
  - Generate new post using RepE-controlled model
- Parallelized for efficiency

### 4. Sentiment Analysis
- Uses CardiffNLP Twitter sentiment model
- Score = P(positive) - P(negative) ∈ [-1, 1]
- Batch processing for speed

### 5. Analysis
- Aggregate by pool type, group size, and coefficient
- Statistical summaries (mean, std, confidence intervals)  
- Multiple visualization modes

## Outputs

### Results Directory (`results/`)
- `sentiment_generation_raw.json` - All generated posts with metadata
- `sentiment_generation_scored.json` - Posts with sentiment scores
- `sentiment_summary.csv` - Aggregated statistics
- Various PNG/PDF/EPS plots

### Key Plots
- **Sentiment vs Group Size** - Shows how feed size affects sentiment under different bias levels
- **Sentiment vs CBI** - Cognitive Bias Index analysis across group sizes
- **Combined Views** - Subset and full analysis side-by-side

## Requirements

- GPU recommended (CPU mode very slow)
- OpenRouter API key for data generation
- ~2-4GB VRAM for Mistral-7B model
- Python packages: torch, transformers, scipy, matplotlib, pandas

## Notes

- Test mode reduces feeds to 2 per size, max size 3 for quick iteration
- Comic Sans MS font auto-detected; falls back to DejaVu Sans
- All plots save in multiple formats (PNG/PDF/EPS) for publication
- Generation and scoring are cached - rerun with `--force` flags to regenerate