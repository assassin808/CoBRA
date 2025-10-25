# 偏差场景与响应生成器

使用OpenRouter API生成偏差场景,使用本地模型生成响应。

## 设置

```bash
pip install -r requirements.txt
```

从 [openrouter.ai](https://openrouter.ai/) 获取OpenRouter API密钥

## 使用方法

### 1. 生成场景
```bash
python scenario_generator.py --api-key YOUR_API_KEY --bias-types authority --num-scenarios 20
```

### 2. 生成响应
```bash
python response_generator.py --input-dir ../data_generated --model "microsoft/DialoGPT-medium"
```

### 3. 与训练管道配合使用
```python
from examples.authority.utils_authority import create_authority_dataset_from_generated
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = create_authority_dataset_from_generated(
    'data_generated/authority_generated_TIMESTAMP_with_responses.json',
    tokenizer
)
```

## 文件

- `scenario_generator.py`: 使用OpenRouter API生成场景
- `response_generator.py`: 使用本地Hugging Face模型生成响应
- `requirements.txt`: 依赖项
