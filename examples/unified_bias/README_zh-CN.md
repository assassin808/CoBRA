# 统一偏差实验框架

在CoBRA项目中运行偏差控制实验的完整指南。

## TL;DR

```bash
# 对权威偏差运行RepE实验
python pipelines.py --bias authority --method repe

# 对确认偏差运行提示基准实验
python pipelines.py --bias confirmation --method prompt_likert

# 对所有偏差使用两种方法运行完整批量实验
python run_batch.py
```

---

## 文件

| 文件 | 用途 |
|------|------|
| `pipelines.py` | **主要实验脚本** - 运行单个偏差/方法实验 |
| `run_batch.py` | **批量执行器** - 对所有偏差运行两种方法 |
| `utils_bias.py` | 数据加载、RepE训练、实验评估的辅助函数 |
| `config.yaml` | 包含所有4种偏差类型的场景定义 |
| `quick_test.py` | 调试时用于快速测试(可选) |
| `clean_outputs.py` | 清理输出目录的实用工具(可选) |

---

## 关键原则

1. **完整性**: 使用 `run_batch.py` 运行所有偏差 × 所有方法
2. **可重现性**: 使用固定随机种子和标准化管道
3. **灵活性**: 使用 `pipelines.py` 定制单个实验

---

## 快速开始

### 导航到此目录
```bash
cd examples/unified_bias
```

### 运行实验

#### 选项A: 单个偏差实验
```bash
# RepE方法
python pipelines.py --bias authority --method repe

# 提示工程方法
python pipelines.py --bias confirmation --method prompt_likert
```

#### 选项B: 完整批量运行
```bash
# 对所有偏差运行两种方法
python run_batch.py

# 跳过RepE,仅运行提示实验
python run_batch.py --skip-repe

# 跳过提示,仅运行RepE实验
python run_batch.py --skip-prompt
```

---

## 可用选项

### pipelines.py 选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `--bias` | 偏差类型 (`authority`, `bandwagon`, `framing`, `confirmation`, `all`) | `authority` |
| `--method` | 方法 (`repe`, `prompt`, `prompt_likert`, `all`) | `repe` |
| `--model` | HuggingFace模型名称 | `mistralai/Mistral-7B-Instruct-v0.3` |
| `--test` | 在测试模式下运行(更快,数据更少) | False |

### run_batch.py 选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `--skip-repe` | 跳过RepE实验 | False |
| `--skip-prompt` | 跳过提示实验 | False |
| `--model` | HuggingFace模型名称 | `mistralai/Mistral-7B-Instruct-v0.3` |

---

## 数据结构

### 输入数据

#### 1. 原始场景数据 (`../../data/`)
场景测试数据:
- `authority/` - Milgram服从、Stanford监狱等场景
- `bandwagon/` - Asch从众、Solomon等场景
- `confirmation/` - Wason、Thaler等场景
- `framing/` - Asian Disease、Survival等场景

#### 2. 生成的训练数据 (`../../data_generated/`)
用于RepE训练的生成对话:
- `authority_generated_*.json`
- `bandwagon_generated_*.json`
- `confirmation_generated_*.json`
- `framing_generated_*.json`

示例格式:
```json
{
  "conversations": [
    {
      "control": "对话文本(无偏差)",
      "treatment": "对话文本(有偏差)",
      "label": 0  // 0 = 对照组, 1 = 实验组
    }
  ]
}
```

### 输出数据

每个实验在 `{bias_type}_plots/` 创建:
- `*.png` - 可视化图表
- `*_plot_data.json` - 原始数据用于重绘
- `*_plot_data.csv` - 用于外部分析的表格数据

---

## 工作原理

### 提示管道 (`--method prompt`)
1. 加载模型和分词器
2. 从原始数据集加载测试场景
3. 运行基于提示的控制实验
4. 保存结果到 `{bias_type}_plots/`

### RepE管道 (`--method repe`)
1. 加载模型和分词器
2. 加载生成的数据用于RepReader训练
3. 训练RepReader检测偏差方向
4. 评估RepReader准确率
5. 从原始数据集加载测试场景
6. 使用训练好的RepReader运行RepE控制实验
7. 保存结果到 `{bias_type}_plots/`

### 输出文件
- **图表**: `{bias_type}_plots/*.png`
- **数据**: `{bias_type}_plots/*_plot_data.json`
- **日志**: 控制台输出

## 示例输出

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
