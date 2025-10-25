# 消融研究

本目录包含用于运行消融研究的脚本,分析不同实验参数对偏差控制的影响。

## 子目录

| 目录 | 用途 |
|-----------|---------|
| `api_experiments/` | **基于API的实验** 用于封闭源码模型 (GPT-4, Claude, Gemini) |
| (当前) | 开源模型的消融研究(角色、温度) |

## 文件

| 文件 | 用途 |
|------|---------|
| `personas_extracted.json` | **角色定义** (25个多样化角色) |
| `run_persona_ablation.py` | 在不同角色间运行实验 |
| `persona_ablation_analysis.py` | 分析角色实验结果 |
| `persona_experiment_helper.py` | 角色实验的辅助工具 |
| `temperature_ablation.py` | 在不同温度间运行实验 |

## 快速开始

### 1. 导航到消融目录
```bash
cd examples/unified_bias/ablation
```

### 2. 运行角色消融

#### 测试模式(推荐首次使用)
```bash
# 使用默认的10个角色运行
python run_persona_ablation.py --bias authority --test

# 使用特定角色运行
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi,Ayesha Khan"
```

#### 完整运行
```bash
# 对权威偏差运行所有角色
python run_persona_ablation.py --bias authority

# 使用自定义模型运行
python run_persona_ablation.py --bias authority --model mistral-7b-local
```

### 3. 分析结果

```bash
# 分析角色实验结果
python persona_ablation_analysis.py \
  --results_dir ./results/mistral-7B-Instruct-v0.3 \
  --output_dir ./persona_analysis
```

### 4. 温度消融

```bash
# 测试模式
python temperature_ablation.py --bias authority --test

# 使用特定温度完整运行
python temperature_ablation.py --bias authority \
  --temperatures 0.1,0.3,0.5,0.7,0.9
```

### 5. 基于API的实验(封闭源码模型)

```bash
# 导航到API实验目录
cd api_experiments

# 使用GPT-4测试
python api_bias.py --bias authority --model gpt4

# 使用Claude测试
python api_bias.py --bias authority --model claude

# 查看完整文档
# api_experiments/README.md
```

## 可用角色

`personas_extracted.json` 文件包含25个多样化角色,包括:

- **学术界**: Adam Smith (哲学家), Giorgio Rossi (数学家), Klaus Mueller (社会学学生)
- **创意界**: Carlos Gomez (诗人), Jennifer Moore (艺术家), Hailey Johnson (作家)
- **专业界**: Yuriko Yamamoto (税务律师), Ryan Park (软件工程师), Latoya Williams (摄影师)
- **服务业**: Arthur Burton (调酒师), Carmen Ortiz (店主), Isabella Rodriguez (咖啡馆老板)
- **学生**: Ayesha Khan (文学), Maria Lopez (物理), Wolfgang Schulz (化学)
- 等等...

## 选项

### run_persona_ablation.py 选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `--bias` | 偏差类型 (`authority`, `bandwagon`, `framing`, `confirmation`, `all`) | `authority` |
| `--personas` | 逗号分隔的角色名称 | 10个默认角色 |
| `--persona-file` | 角色JSON文件路径 | `./personas_extracted.json` |
| `--model` | 使用的模型 | `mistral-7b-local` |
| `--test` | 在测试模式下运行 | False |
| `--permutations` | 排列数 | 1 |
| `--baseline-only` | 仅运行基线(跳过RepE) | False |

### persona_ablation_analysis.py 选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `--results_dir` | 实验结果目录 | `./results/mistral-7B-Instruct-v0.3` |
| `--output_dir` | 分析输出目录 | `./persona_analysis` |
| `--persona_file` | 角色JSON文件路径 | `./personas_extracted.json` |

### temperature_ablation.py 选项

| 选项 | 描述 | 默认值 |
|--------|-------------|---------|
| `--bias` | 偏差类型 | `authority` |
| `--temperatures` | 逗号分隔的温度值 | `0.1,0.2,...,0.9` |
| `--model` | 使用的模型 | `mistral-7b-local` |
| `--test` | 在测试模式下运行 | False |
| `--permutations` | 排列数 | 1 |

## 输出

### 角色消融输出
- **图表**: `{bias_type}_plots_persona_{name}/*.png`
- **数据**: `{bias_type}_plots_persona_{name}/*_plot_data.json`
- **分析**: `persona_analysis/*.png` (对比图)

### 温度消融输出
- **图表**: `{bias_type}_plots_temp{T}/*.png`
- **数据**: `{bias_type}_plots_temp{T}/*_plot_data.json`

## 示例工作流

```bash
cd examples/unified_bias/ablation

# 1. 首先使用3个角色测试
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi,Ayesha Khan"

# 2. 如果成功,使用所有10个默认角色运行完整实验
python run_persona_ablation.py --bias authority

# 3. 分析结果
python persona_ablation_analysis.py \
  --results_dir ./results/mistral-7B-Instruct-v0.3

# 4. 检查输出
ls persona_analysis/
```

## 故障排除

### "personas_extracted.json not found"
**解决方案**: 确保您在 `ablation` 目录中:
```bash
pwd  # 应以 /ablation 结尾
cd examples/unified_bias/ablation
```

### "CUDA out of memory"
**解决方案**: 使用测试模式或减少角色数量:
```bash
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi"
```

### "No results found for analysis"
**解决方案**: 确保实验已成功运行并检查结果目录:
```bash
ls -R results/
python persona_ablation_analysis.py --results_dir ./authority_plots_persona_Adam_Smith
```

## 文件命名约定

角色实验使用此命名模式:
```
{model}_{scenario}_(persona:_{persona_lower})_choice_first_vs_{method}_persona_{PersonaName}_plot_data.json
```

示例:
```
mistral-7B-Instruct-v0.3_milgram_(persona:_adam_smith)_choice_first_vs_prompt_likert_persona_Adam_Smith_plot_data.json
```

## 哲学思考

> "消融研究回答的问题是:
> 什么重要?什么不重要?
> 
> 通过系统地移除或改变一个变量,
> 我们了解它的真正影响。
> 
> 这是科学,不是猜测。"

本目录中的消融研究帮助您理解:
- **角色**: 不同人格如何影响偏差易感性
- **温度**: 生成随机性如何影响控制
- **模型**: 架构如何影响偏差行为(参见父目录中的 `run_batch.py`)
