# 偏差場景與回應生成器

使用OpenRouter API生成偏差場景,使用本地模型生成回應。

## 設定

```bash
pip install -r requirements.txt
```

從 [openrouter.ai](https://openrouter.ai/) 取得OpenRouter API金鑰

## 使用方法

### 1. 生成場景
```bash
python scenario_generator.py --api-key YOUR_API_KEY --bias-types authority --num-scenarios 20
```

### 2. 生成回應
```bash
python response_generator.py --input-dir ../data_generated --model "microsoft/DialoGPT-medium"
```

### 3. 與訓練管線配合使用
```python
from examples.authority.utils_authority import create_authority_dataset_from_generated
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = create_authority_dataset_from_generated(
    'data_generated/authority_generated_TIMESTAMP_with_responses.json',
    tokenizer
)
```

## 檔案

- `scenario_generator.py`: 使用OpenRouter API生成場景
- `response_generator.py`: 使用本地Hugging Face模型生成回應
- `requirements.txt`: 相依套件
