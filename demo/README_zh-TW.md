# 示範: Facebook從眾情緒實驗

本資料夾包含使用現實社交媒體場景進行基於RepE的偏差控制的完整示範。

## 概述

該示範模擬從眾偏差如何影響社交媒體貼文生成中的情緒。使用者在其"動態"中看到不同數量的正面或負面貼文,然後使用具有可控從眾偏差的語言模型生成新貼文。

## 檔案

### 核心實驗
- **`facebook_bandwagon_sentiment_experiment.py`** - 主實驗指令碼
  - 從統一偏差資料集訓練RepE從眾方向
  - 動態發現控制係數
  - 在不同偏差級別和動態大小下生成貼文
  - 評分情緒並生成分析圖表

### 資料生成
- **`generate_facebook_posts.py`** - 建立合成Facebook風格貼文
  - 使用OpenRouter API(GPT-4、Claude、Gemini等)生成多樣化貼文
  - 生成獨立的正面和負面貼文池
  - 可設定數量和選擇性重新生成

### 分析與繪圖
- **`plot_negative_custom.py`** - CBI(認知偏差指數)分析的獨立繪圖
  - 生成Comic Sans MS風格的出版品質圖表
  - 兩個主要圖表: 情緒 vs 群體大小, 情緒 vs CBI
  - 支援不確定性帶(none/std/ci95)

- **`plot_negative_custom_baseline.py`** - 基線偏差級別分析
  - 類似圖表,但用於離散偏差級別而非係數
  - 用於與傳統基於提示的方法比較

### 資料目錄
- **`data/`** - 包含生成的Facebook貼文(由 `generate_facebook_posts.py` 建立)
  - `facebook_positive_posts.json`
  - `facebook_negative_posts.json`

## 快速開始

### 1. 生成資料
首先,使用OpenRouter建立合成Facebook貼文:

```bash
# 設定您的OpenRouter API金鑰
export OPENROUTER_API_KEY="your_key_here"

# 生成100個正面 + 100個負面貼文(預設)
python generate_facebook_posts.py

# 或自訂數量
python generate_facebook_posts.py --pos-total 200 --neg-total 500
```

### 2. 執行實驗
執行主要從眾情緒實驗:

```bash
# 完整實驗(需要GPU)
python facebook_bandwagon_sentiment_experiment.py

# 測試模式(較小規模)
python facebook_bandwagon_sentiment_experiment.py --test-mode

# 從現有生成資料恢復
python facebook_bandwagon_sentiment_experiment.py --reuse-generation

# 僅繪圖模式(重用所有現有資料)
python facebook_bandwagon_sentiment_experiment.py --plot-only
```

### 3. 生成分析圖表
建立出版品質圖表:

```bash
# CBI分析(需要實驗結果)
python plot_negative_custom.py --summary-csv results/sentiment_summary.csv

# 基線偏差級別分析
python plot_negative_custom_baseline.py --summary-csv results/baseline_sentiment_summary.csv
```

## 設定

### 環境變數(.env)
實驗遵循標準控制模組設定:
- `OPENROUTER_API_KEY` - 用於貼文生成
- `REASONING_BATCH_SIZE` - 生成的批次大小
- `MAX_NEW_TOKENS` - 生成貼文的token限制
- `TEMPERATURE` - 取樣溫度

### 命令列選項

**主實驗:**
- `--model` - HuggingFace模型(預設: mistral-7b)
- `--test-mode` - 減少規模以快速測試
- `--max-group-size N` - 動態中的最大貼文數
- `--feeds-per-size N` - 每個大小的隨機動態數
- `--reuse-generation` / `--plot-only` - 跳過昂貴步驟

**貼文生成:**
- `--total N` - 每個類別的貼文數(正面/負面)
- `--pos-total` / `--neg-total` - 獨立計數
- `--models` - 使用的OpenRouter模型
- `--only-positive` / `--only-negative` - 選擇性重新生成

## 管線詳情

### 1. RepE訓練
- 使用現有統一偏差資料集(從眾場景)
- 訓練RepReader識別模型激活中的從眾方向
- 評估各層準確率,選擇前15層進行控制

### 2. 係數發現
- 使用自適應取樣動態找到有效控制範圍
- 兩階段方法: 邊界搜尋 + 基於梯度的細化
- 確保係數產生有意義的行為變化

### 3. 生成
- 對於每個動態大小(0-10)和係數:
  - 從正面/負面池中取樣隨機動態
  - 建構提示: "基於這些貼文: [動態], 你的看法是什麼?"
  - 使用RepE控制的模型生成新貼文
- 平行化以提高效率

### 4. 情緒分析
- 使用CardiffNLP Twitter情緒模型
- 分數 = P(正面) - P(負面) ∈ [-1, 1]
- 批次處理以提高速度

### 5. 分析
- 按池類型、群體大小和係數聚合
- 統計摘要(均值、標準差、置信區間)
- 多種視覺化模式

## 輸出

### 結果目錄(`results/`)
- `sentiment_generation_raw.json` - 所有生成的貼文及詮釋資料
- `sentiment_generation_scored.json` - 帶情緒分數的貼文
- `sentiment_summary.csv` - 聚合統計
- 各種PNG/PDF/EPS圖表

### 關鍵圖表
- **情緒 vs 群體大小** - 顯示動態大小如何在不同偏差級別下影響情緒
- **情緒 vs CBI** - 跨群體大小的認知偏差指數分析
- **組合檢視** - 子集和完整分析並排

## 要求

- 推薦GPU(CPU模式非常慢)
- 用於資料生成的OpenRouter API金鑰
- Mistral-7B模型約2-4GB顯示記憶體
- Python套件: torch, transformers, scipy, matplotlib, pandas

## 注意事項

- 測試模式將動態減少到每個大小2個,最大大小3以快速迭代
- 自動偵測Comic Sans MS字型;回退到DejaVu Sans
- 所有圖表以多種格式(PNG/PDF/EPS)儲存以供發表
- 生成和評分已快取 - 使用 `--force` 旗標重新生成
