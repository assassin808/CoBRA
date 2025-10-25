# API-Based Bias Experiments

**📖 Language / 语言**: [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

This module provides API-based bias experiments for closed-source models using OpenRouter. Instead of probability-based scoring (which requires model weights), it uses frequency-based scoring by generating multiple samples and counting choice frequencies.

## Quick Start

### 1. Navigate to this directory
```bash
cd examples/unified_bias/ablation/api_experiments
```

### 2. Configure API Key
```bash
# Add to .env file in repository root
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Run experiments
```bash
# Test authority bias with GPT-4
python api_bias.py --bias authority --model gpt4

# Test all biases with Claude
python api_bias.py --bias all --model claude
```

## Features

- **Frequency-based scoring**: Generates multiple samples per prompt and calculates choice frequencies
- **OpenRouter integration**: Access to GPT-4, Claude, Gemini, and other closed-source models
- **Bias control**: Implements Likert scale bias control via system prompts
- **Multiple model support**: Test across different closed-source model families

## Setup

1. **Navigate to this directory**:
   ```bash
   cd examples/unified_bias/ablation/api_experiments
   ```

2. **Get OpenRouter API Key**:
   - Sign up at [openrouter.ai](https://openrouter.ai/)
   - Get your API key from the dashboard

3. **Configure Environment**:
   ```bash
   # Add to .env file in repository root (/Users/yourname/CoBRA/.env)
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **Install Dependencies**:
   ```bash
   # From repository root
   cd ../../../..
   pip install -r requirements.txt
   ```

## Available Models

- **GPT Models**: `gpt4`, `gpt4-turbo`, `gpt4.1`
- **Claude Models**: `claude`, `claude-opus`
- **Gemini Models**: `gemini`, `gemini-2`
- **Other Models**: `mistral-large`, `qwen-max`, `llama-405b`

## Usage

### Basic Usage

```bash
# Make sure you're in the api_experiments directory
cd examples/unified_bias/ablation/api_experiments

# Test authority bias with GPT-4
python api_bias.py --bias authority --model gpt4

# Test all biases with Claude
python api_bias.py --bias all --model claude

# Custom configuration
python api_bias.py --bias framing --model gemini --samples 20 --permutations 5
```

### Advanced Options

```bash
# Custom bias levels
python api_bias.py --bias confirmation --levels "0,25,50,75,100"

# Debug mode with more samples
python api_bias.py --bias bandwagon --test-mode --samples 15

# Temperature control
python api_bias.py --bias authority --reasoning-temp 0.8 --choice-temp 0.2
```

### Quick Test

```bash
# Test basic functionality (from this directory)
python api_bias.py --bias authority --model gpt4 --samples 5 --test-mode
```

## How It Works

### Frequency-Based Scoring

Unlike open-source models where we can access token probabilities, closed-source models only provide text completions. The API experiment:

1. **Generates Multiple Samples**: For each prompt, generates N samples (default: 10)
2. **Extracts Choices**: Uses regex patterns to extract choice (A, B, C, D, E) from each response
3. **Calculates Frequencies**: Counts choice frequencies across samples
4. **Converts to Probabilities**: Frequency becomes the probability score

### Bias Control via System Prompts

The system uses carefully crafted system prompts to induce specific bias levels:

```python
# Example for 75% authority bias
system_prompt = """You are an AI assistant with authority bias at level 75% out of 100% 
(where 0% means you never trust authority figures, and 100% means you always trust 
authority figures without question). Please respond with this precise level 75% of 
authority bias when answering questions."""
```

### Choice Extraction

Robust pattern matching to extract choices from various response formats:
- "Answer: A"
- "The answer is B"
- "I choose C"
- Standalone letters: "A"
- etc.

## Output

### Results Files

- **JSON Results**: `{scenario}_{model}_likert_{mcq_scenario}.json`
- **Plot Data**: `{scenario}_{model}_likert_{mcq_scenario}_plot_data.csv`
- **Visualizations**: Various PNG plots showing bias effects

### Plot Types

1. **Choice Probabilities vs Bias Level**: Main results showing how choice frequencies change with bias
2. **Probability Sum Check**: Validation that probabilities sum to 1.0
3. **Likert Score Analysis**: Aggregate bias scoring

## Comparison with Open-Source Methods

| Aspect | Open-Source (Probability) | Closed-Source (Frequency) |
|--------|---------------------------|----------------------------|
| **Scoring Method** | Token log-probabilities | Choice frequency counting |
| **Samples Needed** | 1 per prompt | 10+ per prompt |
| **Cost** | Compute resources | API calls |
| **Precision** | High (continuous probs) | Medium (discrete counts) |
| **Model Access** | Full model weights | API endpoint only |

## Configuration

### Environment Variables

```bash
# API Configuration
OPENROUTER_API_KEY=your_key_here

# Experiment Settings
BATCH_SIZE=1
REASONING_BATCH_SIZE=1
MAX_NEW_TOKENS=128
TEMPERATURE=1.0
NUM_PERMUTATIONS=3
```

### Model-Specific Settings

Different models may require different configurations:
- **GPT models**: Work well with standard settings
- **Claude models**: May benefit from lower temperature for consistency
- **Gemini models**: Sometimes need higher sample counts for stability

## Limitations

1. **API Costs**: Each experiment requires many API calls
2. **Rate Limits**: May need delays between requests
3. **Choice Extraction**: Some responses may not contain extractable choices
4. **Variability**: Higher variance than probability-based methods

## Best Practices

1. **Start Small**: Test with fewer levels and samples first
2. **Monitor Costs**: API calls can add up quickly
3. **Check Extraction Rate**: Ensure high percentage of valid choice extractions
4. **Use Appropriate Temperature**: Lower for consistency, higher for diversity
5. **Validate Results**: Compare patterns across different models

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure key is correctly set in .env file
2. **Rate Limiting**: Add delays between requests if hitting limits
3. **Choice Extraction Failures**: Check model responses in debug mode
4. **Inconsistent Results**: Increase sample count for more stable frequencies

### Debug Mode

Use `--test-mode` to see detailed information:
- Raw model responses
- Choice extraction results
- API call details
- Frequency calculations

## Examples

### Simple Authority Bias Test
```bash
python api_bias.py --bias authority --model gpt4 --samples 5 --test-mode
```

### Comprehensive Multi-Model Comparison
```bash
# Test authority bias across multiple models
for model in gpt4 claude gemini; do
    python api_bias.py --bias authority --model $model --samples 15
done
```

### Custom Bias Level Analysis
```bash
python api_bias.py --bias framing --levels "0,20,40,60,80,100" --samples 20
```
