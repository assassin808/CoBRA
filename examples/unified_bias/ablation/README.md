# Ablation Studies

**📖 Language / 语言**: [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

This directory contains scripts for running ablation studies to analyze the effects of different experimental parameters on bias control.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `api_experiments/` | **API-based experiments** for closed-source models (GPT-4, Claude, Gemini) |
| (current) | Ablation studies for open-source models (persona, temperature) |

## Files

| File | Purpose |
|------|---------|
| `personas_extracted.json` | **Persona definitions** (25 diverse personas) |
| `run_persona_ablation.py` | Run experiments across different personas |
| `persona_ablation_analysis.py` | Analyze persona experiment results |
| `persona_experiment_helper.py` | Helper utilities for persona experiments |
| `temperature_ablation.py` | Run experiments across different temperatures |

## Quick Start

### 1. Navigate to ablation directory
```bash
cd examples/unified_bias/ablation
```

### 2. Run Persona Ablation

#### Test Mode (Recommended First)
```bash
# Run with default 10 personas
python run_persona_ablation.py --bias authority --test

# Run with specific personas
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi,Ayesha Khan"
```

#### Full Run
```bash
# Run all personas on authority bias
python run_persona_ablation.py --bias authority

# Run with custom model
python run_persona_ablation.py --bias authority --model mistral-7b-local
```

### 3. Analyze Results

```bash
# Analyze persona experiment results
python persona_ablation_analysis.py \
  --results_dir ./results/mistral-7B-Instruct-v0.3 \
  --output_dir ./persona_analysis
```

### 4. Temperature Ablation

```bash
# Test mode
python temperature_ablation.py --bias authority --test

# Full run with specific temperatures
python temperature_ablation.py --bias authority \
  --temperatures 0.1,0.3,0.5,0.7,0.9
```

### 5. API-Based Experiments (Closed-Source Models)

```bash
# Navigate to API experiments directory
cd api_experiments

# Test with GPT-4
python api_bias.py --bias authority --model gpt4

# Test with Claude
python api_bias.py --bias authority --model claude

# See api_experiments/README.md for full documentation
```

## Personas Available

The `personas_extracted.json` file contains 25 diverse personas including:

- **Academic**: Adam Smith (philosopher), Giorgio Rossi (mathematician), Klaus Mueller (sociology student)
- **Creative**: Carlos Gomez (poet), Jennifer Moore (artist), Hailey Johnson (writer)
- **Professional**: Yuriko Yamamoto (tax lawyer), Ryan Park (software engineer), Latoya Williams (photographer)
- **Service Industry**: Arthur Burton (bartender), Carmen Ortiz (shopkeeper), Isabella Rodriguez (cafe owner)
- **Students**: Ayesha Khan (literature), Maria Lopez (physics), Wolfgang Schulz (chemistry)
- And more...

## Options

### run_persona_ablation.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bias` | Bias type (`authority`, `bandwagon`, `framing`, `confirmation`, `all`) | `authority` |
| `--personas` | Comma-separated persona names | 10 default personas |
| `--persona-file` | Path to personas JSON | `./personas_extracted.json` |
| `--model` | Model to use | `mistral-7b-local` |
| `--test` | Run in test mode | False |
| `--permutations` | Number of permutations | 1 |
| `--baseline-only` | Run only baselines (skip RepE) | False |

### persona_ablation_analysis.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--results_dir` | Directory with experiment results | `./results/mistral-7B-Instruct-v0.3` |
| `--output_dir` | Output directory for analysis | `./persona_analysis` |
| `--persona_file` | Path to personas JSON | `./personas_extracted.json` |

### temperature_ablation.py Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bias` | Bias type | `authority` |
| `--temperatures` | Comma-separated temperature values | `0.1,0.2,...,0.9` |
| `--model` | Model to use | `mistral-7b-local` |
| `--test` | Run in test mode | False |
| `--permutations` | Number of permutations | 1 |

## Output

### Persona Ablation Output
- **Plots**: `{bias_type}_plots_persona_{name}/*.png`
- **Data**: `{bias_type}_plots_persona_{name}/*_plot_data.json`
- **Analysis**: `persona_analysis/*.png` (comparison plots)

### Temperature Ablation Output
- **Plots**: `{bias_type}_plots_temp{T}/*.png`
- **Data**: `{bias_type}_plots_temp{T}/*_plot_data.json`

## Example Workflow

```bash
cd examples/unified_bias/ablation

# 1. Test with 3 personas first
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi,Ayesha Khan"

# 2. If successful, run full experiment with all 10 default personas
python run_persona_ablation.py --bias authority

# 3. Analyze results
python persona_ablation_analysis.py \
  --results_dir ./results/mistral-7B-Instruct-v0.3

# 4. Check output
ls persona_analysis/
```

## Troubleshooting

### "personas_extracted.json not found"
**Solution**: Make sure you're in the `ablation` directory:
```bash
pwd  # Should end with /ablation
cd examples/unified_bias/ablation
```

### "CUDA out of memory"
**Solution**: Use test mode or reduce number of personas:
```bash
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi"
```

### "No results found for analysis"
**Solution**: Make sure experiments have run successfully and check results directory:
```bash
ls -R results/
python persona_ablation_analysis.py --results_dir ./authority_plots_persona_Adam_Smith
```

## File Naming Convention

Persona experiments use this naming pattern:
```
{model}_{scenario}_(persona:_{persona_lower})_choice_first_vs_{method}_persona_{PersonaName}_plot_data.json
```

Example:
```
mistral-7B-Instruct-v0.3_milgram_(persona:_adam_smith)_choice_first_vs_prompt_likert_persona_Adam_Smith_plot_data.json
```

## Philosophy

> "Ablation studies answer the question: 
> What matters? What doesn't?
> 
> By systematically removing or changing one variable,
> we understand its true impact.
> 
> This is science, not guesswork."

The ablation studies in this directory help you understand:
- **Persona**: How different personalities affect bias susceptibility
- **Temperature**: How generation randomness affects control
- **Model**: How architecture affects bias behavior (see `run_batch.py` in parent directory)
