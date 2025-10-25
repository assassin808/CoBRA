# 演示: Facebook从众情绪实验

本文件夹包含使用现实社交媒体场景进行基于RepE的偏差控制的完整演示。

## 概述

该演示模拟从众偏差如何影响社交媒体帖子生成中的情绪。用户在其"动态"中看到不同数量的正面或负面帖子,然后使用具有可控从众偏差的语言模型生成新帖子。

## 文件

### 核心实验
- **`facebook_bandwagon_sentiment_experiment.py`** - 主实验脚本
  - 从统一偏差数据集训练RepE从众方向
  - 动态发现控制系数
  - 在不同偏差级别和动态大小下生成帖子
  - 评分情绪并生成分析图表

### 数据生成
- **`generate_facebook_posts.py`** - 创建合成Facebook风格帖子
  - 使用OpenRouter API(GPT-4、Claude、Gemini等)生成多样化帖子
  - 生成独立的正面和负面帖子池
  - 可配置数量和选择性重新生成

### 分析与绘图
- **`plot_negative_custom.py`** - CBI(认知偏差指数)分析的独立绘图
  - 生成Comic Sans MS风格的出版质量图表
  - 两个主要图表: 情绪 vs 群体大小, 情绪 vs CBI
  - 支持不确定性带(none/std/ci95)

- **`plot_negative_custom_baseline.py`** - 基线偏差级别分析
  - 类似图表,但用于离散偏差级别而非系数
  - 用于与传统基于提示的方法比较

### 数据目录
- **`data/`** - 包含生成的Facebook帖子(由 `generate_facebook_posts.py` 创建)
  - `facebook_positive_posts.json`
  - `facebook_negative_posts.json`

## 快速开始

### 1. 生成数据
首先,使用OpenRouter创建合成Facebook帖子:

```bash
# 设置您的OpenRouter API密钥
export OPENROUTER_API_KEY="your_key_here"

# 生成100个正面 + 100个负面帖子(默认)
python generate_facebook_posts.py

# 或自定义数量
python generate_facebook_posts.py --pos-total 200 --neg-total 500
```

### 2. 运行实验
运行主要从众情绪实验:

```bash
# 完整实验(需要GPU)
python facebook_bandwagon_sentiment_experiment.py

# 测试模式(较小规模)
python facebook_bandwagon_sentiment_experiment.py --test-mode

# 从现有生成数据恢复
python facebook_bandwagon_sentiment_experiment.py --reuse-generation

# 仅绘图模式(重用所有现有数据)
python facebook_bandwagon_sentiment_experiment.py --plot-only
```

### 3. 生成分析图表
创建出版质量图表:

```bash
# CBI分析(需要实验结果)
python plot_negative_custom.py --summary-csv results/sentiment_summary.csv

# 基线偏差级别分析
python plot_negative_custom_baseline.py --summary-csv results/baseline_sentiment_summary.csv
```

## 配置

### 环境变量(.env)
实验遵循标准控制模块设置:
- `OPENROUTER_API_KEY` - 用于帖子生成
- `REASONING_BATCH_SIZE` - 生成的批量大小
- `MAX_NEW_TOKENS` - 生成帖子的token限制
- `TEMPERATURE` - 采样温度

### 命令行选项

**主实验:**
- `--model` - HuggingFace模型(默认: mistral-7b)
- `--test-mode` - 减少规模以快速测试
- `--max-group-size N` - 动态中的最大帖子数
- `--feeds-per-size N` - 每个大小的随机动态数
- `--reuse-generation` / `--plot-only` - 跳过昂贵步骤

**帖子生成:**
- `--total N` - 每个类别的帖子数(正面/负面)
- `--pos-total` / `--neg-total` - 独立计数
- `--models` - 使用的OpenRouter模型
- `--only-positive` / `--only-negative` - 选择性重新生成

## 管道详情

### 1. RepE训练
- 使用现有统一偏差数据集(从众场景)
- 训练RepReader识别模型激活中的从众方向
- 评估各层准确率,选择前15层进行控制

### 2. 系数发现
- 使用自适应采样动态找到有效控制范围
- 两阶段方法: 边界搜索 + 基于梯度的细化
- 确保系数产生有意义的行为变化

### 3. 生成
- 对于每个动态大小(0-10)和系数:
  - 从正面/负面池中采样随机动态
  - 构建提示: "基于这些帖子: [动态], 你的看法是什么?"
  - 使用RepE控制的模型生成新帖子
- 并行化以提高效率

### 4. 情绪分析
- 使用CardiffNLP Twitter情绪模型
- 分数 = P(正面) - P(负面) ∈ [-1, 1]
- 批量处理以提高速度

### 5. 分析
- 按池类型、群体大小和系数聚合
- 统计摘要(均值、标准差、置信区间)
- 多种可视化模式

## 输出

### 结果目录(`results/`)
- `sentiment_generation_raw.json` - 所有生成的帖子及元数据
- `sentiment_generation_scored.json` - 带情绪分数的帖子
- `sentiment_summary.csv` - 聚合统计
- 各种PNG/PDF/EPS图表

### 关键图表
- **情绪 vs 群体大小** - 显示动态大小如何在不同偏差级别下影响情绪
- **情绪 vs CBI** - 跨群体大小的认知偏差指数分析
- **组合视图** - 子集和完整分析并排

## 要求

- 推荐GPU(CPU模式非常慢)
- 用于数据生成的OpenRouter API密钥
- Mistral-7B模型约2-4GB显存
- Python包: torch, transformers, scipy, matplotlib, pandas

## 注意事项

- 测试模式将动态减少到每个大小2个,最大大小3以快速迭代
- 自动检测Comic Sans MS字体;回退到DejaVu Sans
- 所有图表以多种格式(PNG/PDF/EPS)保存以供发表
- 生成和评分已缓存 - 使用 `--force` 标志重新生成
