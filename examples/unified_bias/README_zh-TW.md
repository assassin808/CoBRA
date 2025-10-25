# 統一偏差實驗框架

在CoBRA專案中執行偏差控制實驗的完整指南。

## TL;DR

```bash
# 對權威偏差執行RepE實驗
python pipelines.py --bias authority --method repe

# 對確認偏差執行提示基準實驗
python pipelines.py --bias confirmation --method prompt_likert

# 對所有偏差使用兩種方法執行完整批次實驗
python run_batch.py
```

---

## 檔案

| 檔案 | 用途 |
|------|------|
| `pipelines.py` | **主要實驗指令碼** - 執行單個偏差/方法實驗 |
| `run_batch.py` | **批次執行器** - 對所有偏差執行兩種方法 |
| `utils_bias.py` | 資料載入、RepE訓練、實驗評估的輔助函式 |
| `config.yaml` | 包含所有4種偏差類型的場景定義 |
| `quick_test.py` | 除錯時用於快速測試(可選) |
| `clean_outputs.py` | 清理輸出目錄的實用工具(可選) |

---

## 關鍵原則

1. **完整性**: 使用 `run_batch.py` 執行所有偏差 × 所有方法
2. **可重現性**: 使用固定隨機種子和標準化管線
3. **靈活性**: 使用 `pipelines.py` 定製單個實驗

---

## 快速開始

### 導航到此目錄
```bash
cd examples/unified_bias
```

### 執行實驗

#### 選項A: 單個偏差實驗
```bash
# RepE方法
python pipelines.py --bias authority --method repe

# 提示工程方法
python pipelines.py --bias confirmation --method prompt_likert
```

#### 選項B: 完整批次執行
```bash
# 對所有偏差執行兩種方法
python run_batch.py

# 跳過RepE,僅執行提示實驗
python run_batch.py --skip-repe

# 跳過提示,僅執行RepE實驗
python run_batch.py --skip-prompt
```

---

## 可用選項

### pipelines.py 選項

| 選項 | 描述 | 預設值 |
|--------|-------------|---------|
| `--bias` | 偏差類型 (`authority`, `bandwagon`, `framing`, `confirmation`, `all`) | `authority` |
| `--method` | 方法 (`repe`, `prompt`, `prompt_likert`, `all`) | `repe` |
| `--model` | HuggingFace模型名稱 | `mistralai/Mistral-7B-Instruct-v0.3` |
| `--test` | 在測試模式下執行(更快,資料更少) | False |

### run_batch.py 選項

| 選項 | 描述 | 預設值 |
|--------|-------------|---------|
| `--skip-repe` | 跳過RepE實驗 | False |
| `--skip-prompt` | 跳過提示實驗 | False |
| `--model` | HuggingFace模型名稱 | `mistralai/Mistral-7B-Instruct-v0.3` |

---

## 資料結構

### 輸入資料

#### 1. 原始場景資料 (`../../data/`)
場景測試資料:
- `authority/` - Milgram服從、Stanford監獄等場景
- `bandwagon/` - Asch從眾、Solomon等場景
- `confirmation/` - Wason、Thaler等場景
- `framing/` - Asian Disease、Survival等場景

#### 2. 生成的訓練資料 (`../../data_generated/`)
用於RepE訓練的生成對話:
- `authority_generated_*.json`
- `bandwagon_generated_*.json`
- `confirmation_generated_*.json`
- `framing_generated_*.json`

範例格式:
```json
{
  "conversations": [
    {
      "control": "對話文字(無偏差)",
      "treatment": "對話文字(有偏差)",
      "label": 0  // 0 = 對照組, 1 = 實驗組
    }
  ]
}
```

### 輸出資料

每個實驗在 `{bias_type}_plots/` 建立:
- `*.png` - 視覺化圖表
- `*_plot_data.json` - 原始資料用於重繪
- `*_plot_data.csv` - 用於外部分析的表格資料

---

## 工作原理

### 提示管線 (`--method prompt`)
1. 載入模型和分詞器
2. 從原始資料集載入測試場景
3. 執行基於提示的控制實驗
4. 儲存結果到 `{bias_type}_plots/`

### RepE管線 (`--method repe`)
1. 載入模型和分詞器
2. 載入生成的資料用於RepReader訓練
3. 訓練RepReader偵測偏差方向
4. 評估RepReader準確率
5. 從原始資料集載入測試場景
6. 使用訓練好的RepReader執行RepE控制實驗
7. 儲存結果到 `{bias_type}_plots/`

### 輸出檔案
- **圖表**: `{bias_type}_plots/*.png`
- **資料**: `{bias_type}_plots/*_plot_data.json`
- **日誌**: 控制台輸出

## 範例輸出

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
