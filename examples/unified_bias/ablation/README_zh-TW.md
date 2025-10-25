# 消融研究

本目錄包含用於執行消融研究的指令碼,分析不同實驗參數對偏差控制的影響。

## 子目錄

| 目錄 | 用途 |
|-----------|---------|
| `api_experiments/` | **基於API的實驗** 用於封閉原始碼模型 (GPT-4, Claude, Gemini) |
| (當前) | 開源模型的消融研究(角色、溫度) |

## 檔案

| 檔案 | 用途 |
|------|---------|
| `personas_extracted.json` | **角色定義** (25個多樣化角色) |
| `run_persona_ablation.py` | 在不同角色間執行實驗 |
| `persona_ablation_analysis.py` | 分析角色實驗結果 |
| `persona_experiment_helper.py` | 角色實驗的輔助工具 |
| `temperature_ablation.py` | 在不同溫度間執行實驗 |

## 快速開始

### 1. 導航到消融目錄
```bash
cd examples/unified_bias/ablation
```

### 2. 執行角色消融

#### 測試模式(推薦首次使用)
```bash
# 使用預設的10個角色執行
python run_persona_ablation.py --bias authority --test

# 使用特定角色執行
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi,Ayesha Khan"
```

#### 完整執行
```bash
# 對權威偏差執行所有角色
python run_persona_ablation.py --bias authority

# 使用自訂模型執行
python run_persona_ablation.py --bias authority --model mistral-7b-local
```

### 3. 分析結果

```bash
# 分析角色實驗結果
python persona_ablation_analysis.py \
  --results_dir ./results/mistral-7B-Instruct-v0.3 \
  --output_dir ./persona_analysis
```

### 4. 溫度消融

```bash
# 測試模式
python temperature_ablation.py --bias authority --test

# 使用特定溫度完整執行
python temperature_ablation.py --bias authority \
  --temperatures 0.1,0.3,0.5,0.7,0.9
```

### 5. 基於API的實驗(封閉原始碼模型)

```bash
# 導航到API實驗目錄
cd api_experiments

# 使用GPT-4測試
python api_bias.py --bias authority --model gpt4

# 使用Claude測試
python api_bias.py --bias authority --model claude

# 檢視完整文件
# api_experiments/README.md
```

## 可用角色

`personas_extracted.json` 檔案包含25個多樣化角色,包括:

- **學術界**: Adam Smith (哲學家), Giorgio Rossi (數學家), Klaus Mueller (社會學學生)
- **創意界**: Carlos Gomez (詩人), Jennifer Moore (藝術家), Hailey Johnson (作家)
- **專業界**: Yuriko Yamamoto (稅務律師), Ryan Park (軟體工程師), Latoya Williams (攝影師)
- **服務業**: Arthur Burton (調酒師), Carmen Ortiz (店主), Isabella Rodriguez (咖啡館老闆)
- **學生**: Ayesha Khan (文學), Maria Lopez (物理), Wolfgang Schulz (化學)
- 等等...

## 選項

### run_persona_ablation.py 選項

| 選項 | 描述 | 預設值 |
|--------|-------------|---------|
| `--bias` | 偏差類型 (`authority`, `bandwagon`, `framing`, `confirmation`, `all`) | `authority` |
| `--personas` | 逗號分隔的角色名稱 | 10個預設角色 |
| `--persona-file` | 角色JSON檔案路徑 | `./personas_extracted.json` |
| `--model` | 使用的模型 | `mistral-7b-local` |
| `--test` | 在測試模式下執行 | False |
| `--permutations` | 排列數 | 1 |
| `--baseline-only` | 僅執行基線(跳過RepE) | False |

### persona_ablation_analysis.py 選項

| 選項 | 描述 | 預設值 |
|--------|-------------|---------|
| `--results_dir` | 實驗結果目錄 | `./results/mistral-7B-Instruct-v0.3` |
| `--output_dir` | 分析輸出目錄 | `./persona_analysis` |
| `--persona_file` | 角色JSON檔案路徑 | `./personas_extracted.json` |

### temperature_ablation.py 選項

| 選項 | 描述 | 預設值 |
|--------|-------------|---------|
| `--bias` | 偏差類型 | `authority` |
| `--temperatures` | 逗號分隔的溫度值 | `0.1,0.2,...,0.9` |
| `--model` | 使用的模型 | `mistral-7b-local` |
| `--test` | 在測試模式下執行 | False |
| `--permutations` | 排列數 | 1 |

## 輸出

### 角色消融輸出
- **圖表**: `{bias_type}_plots_persona_{name}/*.png`
- **資料**: `{bias_type}_plots_persona_{name}/*_plot_data.json`
- **分析**: `persona_analysis/*.png` (對比圖)

### 溫度消融輸出
- **圖表**: `{bias_type}_plots_temp{T}/*.png`
- **資料**: `{bias_type}_plots_temp{T}/*_plot_data.json`

## 範例工作流

```bash
cd examples/unified_bias/ablation

# 1. 首先使用3個角色測試
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi,Ayesha Khan"

# 2. 如果成功,使用所有10個預設角色執行完整實驗
python run_persona_ablation.py --bias authority

# 3. 分析結果
python persona_ablation_analysis.py \
  --results_dir ./results/mistral-7B-Instruct-v0.3

# 4. 檢查輸出
ls persona_analysis/
```

## 故障排除

### "personas_extracted.json not found"
**解決方案**: 確保您在 `ablation` 目錄中:
```bash
pwd  # 應以 /ablation 結尾
cd examples/unified_bias/ablation
```

### "CUDA out of memory"
**解決方案**: 使用測試模式或減少角色數量:
```bash
python run_persona_ablation.py --bias authority --test \
  --personas "Adam Smith,Giorgio Rossi"
```

### "No results found for analysis"
**解決方案**: 確保實驗已成功執行並檢查結果目錄:
```bash
ls -R results/
python persona_ablation_analysis.py --results_dir ./authority_plots_persona_Adam_Smith
```

## 檔案命名約定

角色實驗使用此命名模式:
```
{model}_{scenario}_(persona:_{persona_lower})_choice_first_vs_{method}_persona_{PersonaName}_plot_data.json
```

範例:
```
mistral-7B-Instruct-v0.3_milgram_(persona:_adam_smith)_choice_first_vs_prompt_likert_persona_Adam_Smith_plot_data.json
```

## 哲學思考

> "消融研究回答的問題是:
> 什麼重要?什麼不重要?
> 
> 透過系統地移除或改變一個變數,
> 我們瞭解它的真正影響。
> 
> 這是科學,不是猜測。"

本目錄中的消融研究幫助您理解:
- **角色**: 不同人格如何影響偏差易感性
- **溫度**: 生成隨機性如何影響控制
- **模型**: 架構如何影響偏差行為(參見父目錄中的 `run_batch.py`)
