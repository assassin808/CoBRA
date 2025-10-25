# Unified Bias Experiment Framework

**📖 Language / 语言**: [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

Run bias control experiments for authority, bandwagon, framing, and confirmation biases.

## TL;DR

```bash
cd examples/unified_bias

# Run prompt-based experiment (fast)
python run_pipelines.py --bias authority --method prompt --test

# Run RepE experiment (requires training)
python run_pipelines.py --bias authority --method repe --test

# Run all models in batch
python run_batch.py
```

---

## Files

| File | Purpose |
|------|---------|
| `run_pipelines.py` | **Main entry point** - Run single experiments |
| `pipelines.py` | Core logic for prompt/RepE pipelines |
| `run_batch.py` | Batch runner for multiple models |
| `utils_bias.py` | Data management and utilities |
| `bias.py` | ⚠️ Deprecated - use `run_pipelines.py` |

### Key Principles
- **Separation of concerns**: Prompt and RepE are separate pipelines
- **Reusable training**: RepReader trained once, used for all experiments
- **Clean data flow**: Generated data for training, original data for testing

## Quick Start

### 1. Navigate to the correct directory
```bash
cd examples/unified_bias
```

### 2. Run Experiments

#### Prompt-based Experiments
```bash
# Basic usage
python run_pipelines.py --bias authority --method prompt

# Test mode (faster, fewer scenarios)
python run_pipelines.py --bias bandwagon --method prompt --test

# With custom temperature
python run_pipelines.py --bias framing --method prompt --temp 0.7
```

#### RepE Experiments
```bash
# Basic usage
python run_pipelines.py --bias authority --method repe

# With specific operators
python run_pipelines.py --bias confirmation --method repe --operators linear_comb,projection

# Test mode
python run_pipelines.py --bias framing --method repe --test
```

### 3. Available Options

| Option | Values | Description |
|--------|--------|-------------|
| `--bias` | `authority`, `bandwagon`, `framing`, `confirmation` | Which bias type to test |
| `--method` | `prompt`, `repe` | Experiment method |
| `--test` | flag | Run in test mode (4 scenarios only) |
| `--temp` | float (e.g., `0.7`) | Temperature for generation |
| `--model` | model name | Specific model to use |
| `--operators` | comma-separated | RepE operators (e.g., `linear_comb,projection`) |


## Data Structure

### Original Datasets (Testing)
Located in `../../data/`

| Bias Type | Experiments |
|-----------|-------------|
| Authority | `authority_MilgramS.json` (milgram)<br>`authority_StanPri.json` (stanford) |
| Bandwagon | `bandwagon_Asch.json` (asch)<br>`bandwagon_Hotel.json` (hotel) |
| Framing | `framing_Asian.json` (asian)<br>`framing_Invest.json` (invest) |
| Confirmation | `confirmation_BiasInfo.json` (bias_info)<br>`confirmation_Wason.json` (wason) |

### Generated Datasets (Training)
Located in `../../data_generated/`

- `authority_generated_20250810_160938.json`
- `bandwagon_generated.json`
- `framing_generated.json`
- `confirmation_generated.json`

## How It Works

### Prompt Pipeline (`--method prompt`)
1. Load model and tokenizer
2. Load test scenarios from original datasets
3. Run prompt-based control experiments
4. Save results to `{bias_type}_plots/`

### RepE Pipeline (`--method repe`)
1. Load model and tokenizer
2. Load generated data for RepReader training
3. Train RepReader to detect bias directions
4. Evaluate RepReader accuracy
5. Load test scenarios from original datasets
6. Run RepE control experiments with trained RepReader
7. Save results to `{bias_type}_plots/`

### Output Files
- **Plots**: `{bias_type}_plots/*.png`
- **Data**: `{bias_type}_plots/*_plot_data.json`
- **Logs**: Console output

## Example Output

```bash
$ cd examples/unified_bias
$ python run_pipelines.py --bias authority --method prompt --test

=== PROMPT PIPELINE :: authority ===
--- 1. Loading Model and Tokenizer ---
Model loaded on device: cuda

--- Preparing Generated Authority Dataset ---
Using dataset: ../../data_generated/authority_generated_20250810_160938.json
Training data size: 512, Test data size: 256

--- Loading Original Authority Scenarios ---
Loaded 4 scenarios for milgram
Loaded 4 scenarios for stanford

--- Prompt Control :: Authority - Milgram ---
Running experiments...
✓ Results saved

--- Prompt Control :: Authority - Stanford ---
Running experiments...
✓ Results saved

[Prompt] Results saved in authority_plots/
```
