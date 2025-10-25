# 基于API的偏差实验

本模块为使用OpenRouter的封闭源码模型提供基于API的偏差实验。与基于概率的评分(需要模型权重)不同,它通过生成多个样本并计算选择频率来使用基于频率的评分。

## 快速开始

### 1. 导航到此目录
```bash
cd examples/unified_bias/ablation/api_experiments
```

### 2. 配置API密钥
```bash
# 在仓库根目录的.env文件中添加
OPENROUTER_API_KEY=your_api_key_here
```

### 3. 运行实验
```bash
# 使用GPT-4测试权威偏差
python api_bias.py --bias authority --model gpt4

# 使用Claude测试所有偏差
python api_bias.py --bias all --model claude
```

## 特性

- **基于频率的评分**: 为每个提示生成多个样本并计算选择频率
- **OpenRouter集成**: 访问GPT-4、Claude、Gemini和其他封闭源码模型
- **偏差控制**: 通过系统提示实现Likert量表偏差控制
- **多模型支持**: 跨不同封闭源码模型家族测试

## 设置

1. **导航到此目录**:
   ```bash
   cd examples/unified_bias/ablation/api_experiments
   ```

2. **获取OpenRouter API密钥**:
   - 在 [openrouter.ai](https://openrouter.ai/) 注册
   - 从仪表板获取您的API密钥

3. **配置环境**:
   ```bash
   # 在仓库根目录的.env文件中添加 (/Users/yourname/CoBRA/.env)
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **安装依赖**:
   ```bash
   # 从仓库根目录
   cd ../../../..
   pip install -r requirements.txt
   ```

## 可用模型

- **GPT模型**: `gpt4`, `gpt4-turbo`, `gpt4.1`
- **Claude模型**: `claude`, `claude-opus`
- **Gemini模型**: `gemini`, `gemini-2`
- **其他模型**: `mistral-large`, `qwen-max`, `llama-405b`

## 使用方法

### 基本用法

```bash
# 确保您在api_experiments目录中
cd examples/unified_bias/ablation/api_experiments

# 使用GPT-4测试权威偏差
python api_bias.py --bias authority --model gpt4

# 使用Claude测试所有偏差
python api_bias.py --bias all --model claude

# 自定义配置
python api_bias.py --bias framing --model gemini --samples 20 --permutations 5
```

### 高级选项

```bash
# 自定义偏差级别
python api_bias.py --bias confirmation --levels "0,25,50,75,100"

# 使用更多样本的调试模式
python api_bias.py --bias bandwagon --test-mode --samples 15

# 温度控制
python api_bias.py --bias authority --reasoning-temp 0.8 --choice-temp 0.2
```

### 快速测试

```bash
# 测试基本功能(从此目录)
python api_bias.py --bias authority --model gpt4 --samples 5 --test-mode
```

## 工作原理

### 基于频率的评分

与我们可以访问token概率的开源模型不同,封闭源码模型仅提供文本补全。API实验:

1. **生成多个样本**: 为每个提示生成N个样本(默认: 10)
2. **提取选择**: 使用正则表达式模式从每个响应中提取选择(A、B、C、D、E)
3. **计算频率**: 统计样本中的选择频率
4. **转换为概率**: 频率成为概率分数

### 通过系统提示进行偏差控制

系统使用精心设计的系统提示来诱导特定的偏差级别:

```python
# 75%权威偏差的示例
system_prompt = """您是一个权威偏差级别为75%(满分100%)的AI助手
(其中0%表示您从不信任权威人物,100%表示您毫无疑问地总是信任
权威人物)。在回答问题时,请以这个精确的75%权威偏差级别进行响应。"""
```

### 选择提取

使用稳健的模式匹配从各种响应格式中提取选择:
- "Answer: A"
- "The answer is B"
- "I choose C"
- 独立字母: "A"
- 等等

## 输出

### 结果文件

- **JSON结果**: `{scenario}_{model}_likert_{mcq_scenario}.json`
- **绘图数据**: `{scenario}_{model}_likert_{mcq_scenario}_plot_data.csv`
- **可视化**: 显示偏差效果的各种PNG图表

### 图表类型

1. **选择概率 vs 偏差级别**: 显示选择频率如何随偏差变化的主要结果
2. **概率总和检查**: 验证概率总和为1.0
3. **Likert分数分析**: 聚合偏差评分

## 与开源方法的比较

| 方面 | 开源(概率) | 封闭源码(频率) |
|--------|---------------------------|----------------------------|
| **评分方法** | Token对数概率 | 选择频率计数 |
| **所需样本** | 每个提示1个 | 每个提示10+个 |
| **成本** | 计算资源 | API调用 |
| **精度** | 高(连续概率) | 中等(离散计数) |
| **模型访问** | 完整模型权重 | 仅API端点 |

## 配置

### 环境变量

```bash
# API配置
OPENROUTER_API_KEY=your_key_here

# 实验设置
BATCH_SIZE=1
REASONING_BATCH_SIZE=1
MAX_NEW_TOKENS=128
TEMPERATURE=1.0
NUM_PERMUTATIONS=3
```

### 模型特定设置

不同模型可能需要不同配置:
- **GPT模型**: 标准设置效果良好
- **Claude模型**: 可能受益于较低温度以保持一致性
- **Gemini模型**: 有时需要更高样本数以保持稳定性

## 限制

1. **API成本**: 每个实验需要许多API调用
2. **速率限制**: 请求之间可能需要延迟
3. **选择提取**: 某些响应可能不包含可提取的选择
4. **变异性**: 比基于概率的方法有更高方差

## 最佳实践

1. **从小开始**: 首先使用更少的级别和样本测试
2. **监控成本**: API调用可能会迅速累积
3. **检查提取率**: 确保有效选择提取的高百分比
4. **使用适当温度**: 较低温度用于一致性,较高温度用于多样性
5. **验证结果**: 比较不同模型间的模式

## 故障排除

### 常见问题

1. **API密钥错误**: 确保密钥在.env文件中正确设置
2. **速率限制**: 如果达到限制,在请求之间添加延迟
3. **选择提取失败**: 在调试模式下检查模型响应
4. **结果不一致**: 增加样本数以获得更稳定的频率

### 调试模式

使用 `--test-mode` 查看详细信息:
- 原始模型响应
- 选择提取结果
- API调用详情
- 频率计算

## 示例

### 简单权威偏差测试
```bash
python api_bias.py --bias authority --model gpt4 --samples 5 --test-mode
```

### 全面多模型比较
```bash
# 跨多个模型测试权威偏差
for model in gpt4 claude gemini; do
    python api_bias.py --bias authority --model $model --samples 15
done
```

### 自定义偏差级别分析
```bash
python api_bias.py --bias framing --levels "0,20,40,60,80,100" --samples 20
```
