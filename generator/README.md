# Bias Scenario & Response Generator

**📖 Language / 语言**: [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

Generate bias scenarios using OpenRouter API and responses using local models.

## Setup

```bash
pip install -r requirements.txt
```

Get OpenRouter API key from [openrouter.ai](https://openrouter.ai/)

## Usage

### 1. Generate Scenarios
```bash
python scenario_generator.py --api-key YOUR_API_KEY --bias-types authority --num-scenarios 20
```

### 2. Generate Responses  
```bash
python response_generator.py --input-dir ../data_generated --model "microsoft/DialoGPT-medium"
```

### 3. Use with Training Pipeline
```python
from examples.authority.utils_authority import create_authority_dataset_from_generated
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = create_authority_dataset_from_generated(
    'data_generated/authority_generated_TIMESTAMP_with_responses.json',
    tokenizer
)
```

## Files

- `scenario_generator.py`: Generate scenarios using OpenRouter API
- `response_generator.py`: Generate responses using local Hugging Face models
- `requirements.txt`: Dependencies
