# 基於API的偏差實驗

本模組為使用OpenRouter的封閉原始碼模型提供基於API的偏差實驗。與基於機率的評分(需要模型權重)不同,它透過生成多個樣本並計算選擇頻率來使用基於頻率的評分。

## 快速開始

### 1. 導航到此目錄
```bash
cd examples/unified_bias/ablation/api_experiments
```

### 2. 設定API金鑰
```bash
# 在倉儲根目錄的.env檔案中新增
OPENROUTER_API_KEY=your_api_key_here
```

### 3. 執行實驗
```bash
# 使用GPT-4測試權威偏差
python api_bias.py --bias authority --model gpt4

# 使用Claude測試所有偏差
python api_bias.py --bias all --model claude
```

## 特性

- **基於頻率的評分**: 為每個提示生成多個樣本並計算選擇頻率
- **OpenRouter整合**: 存取GPT-4、Claude、Gemini和其他封閉原始碼模型
- **偏差控制**: 透過系統提示實現Likert量表偏差控制
- **多模型支援**: 跨不同封閉原始碼模型家族測試

## 設定

1. **導航到此目錄**:
   ```bash
   cd examples/unified_bias/ablation/api_experiments
   ```

2. **取得OpenRouter API金鑰**:
   - 在 [openrouter.ai](https://openrouter.ai/) 註冊
   - 從儀表板取得您的API金鑰

3. **設定環境**:
   ```bash
   # 在倉儲根目錄的.env檔案中新增 (/Users/yourname/CoBRA/.env)
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **安裝相依套件**:
   ```bash
   # 從倉儲根目錄
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
# 確保您在api_experiments目錄中
cd examples/unified_bias/ablation/api_experiments

# 使用GPT-4測試權威偏差
python api_bias.py --bias authority --model gpt4

# 使用Claude測試所有偏差
python api_bias.py --bias all --model claude

# 自訂設定
python api_bias.py --bias framing --model gemini --samples 20 --permutations 5
```

### 進階選項

```bash
# 自訂偏差級別
python api_bias.py --bias confirmation --levels "0,25,50,75,100"

# 使用更多樣本的除錯模式
python api_bias.py --bias bandwagon --test-mode --samples 15

# 溫度控制
python api_bias.py --bias authority --reasoning-temp 0.8 --choice-temp 0.2
```

### 快速測試

```bash
# 測試基本功能(從此目錄)
python api_bias.py --bias authority --model gpt4 --samples 5 --test-mode
```

## 工作原理

### 基於頻率的評分

與我們可以存取token機率的開源模型不同,封閉原始碼模型僅提供文字補全。API實驗:

1. **生成多個樣本**: 為每個提示生成N個樣本(預設: 10)
2. **提取選擇**: 使用正規表示式模式從每個回應中提取選擇(A、B、C、D、E)
3. **計算頻率**: 統計樣本中的選擇頻率
4. **轉換為機率**: 頻率成為機率分數

### 透過系統提示進行偏差控制

系統使用精心設計的系統提示來誘導特定的偏差級別:

```python
# 75%權威偏差的範例
system_prompt = """您是一個權威偏差級別為75%(滿分100%)的AI助手
(其中0%表示您從不信任權威人物,100%表示您毫無疑問地總是信任
權威人物)。在回答問題時,請以這個精確的75%權威偏差級別進行回應。"""
```

### 選擇提取

使用穩健的模式匹配從各種回應格式中提取選擇:
- "Answer: A"
- "The answer is B"
- "I choose C"
- 獨立字母: "A"
- 等等

## 輸出

### 結果檔案

- **JSON結果**: `{scenario}_{model}_likert_{mcq_scenario}.json`
- **繪圖資料**: `{scenario}_{model}_likert_{mcq_scenario}_plot_data.csv`
- **視覺化**: 顯示偏差效果的各種PNG圖表

### 圖表類型

1. **選擇機率 vs 偏差級別**: 顯示選擇頻率如何隨偏差變化的主要結果
2. **機率總和檢查**: 驗證機率總和為1.0
3. **Likert分數分析**: 聚合偏差評分

## 與開源方法的比較

| 方面 | 開源(機率) | 封閉原始碼(頻率) |
|--------|---------------------------|----------------------------|
| **評分方法** | Token對數機率 | 選擇頻率計數 |
| **所需樣本** | 每個提示1個 | 每個提示10+個 |
| **成本** | 計算資源 | API呼叫 |
| **精度** | 高(連續機率) | 中等(離散計數) |
| **模型存取** | 完整模型權重 | 僅API端點 |

## 設定

### 環境變數

```bash
# API設定
OPENROUTER_API_KEY=your_key_here

# 實驗設定
BATCH_SIZE=1
REASONING_BATCH_SIZE=1
MAX_NEW_TOKENS=128
TEMPERATURE=1.0
NUM_PERMUTATIONS=3
```

### 模型特定設定

不同模型可能需要不同設定:
- **GPT模型**: 標準設定效果良好
- **Claude模型**: 可能受益於較低溫度以保持一致性
- **Gemini模型**: 有時需要更高樣本數以保持穩定性

## 限制

1. **API成本**: 每個實驗需要許多API呼叫
2. **速率限制**: 請求之間可能需要延遲
3. **選擇提取**: 某些回應可能不包含可提取的選擇
4. **變異性**: 比基於機率的方法有更高變異數

## 最佳實踐

1. **從小開始**: 首先使用更少的級別和樣本測試
2. **監控成本**: API呼叫可能會迅速累積
3. **檢查提取率**: 確保有效選擇提取的高百分比
4. **使用適當溫度**: 較低溫度用於一致性,較高溫度用於多樣性
5. **驗證結果**: 比較不同模型間的模式

## 故障排除

### 常見問題

1. **API金鑰錯誤**: 確保金鑰在.env檔案中正確設定
2. **速率限制**: 如果達到限制,在請求之間新增延遲
3. **選擇提取失敗**: 在除錯模式下檢查模型回應
4. **結果不一致**: 增加樣本數以獲得更穩定的頻率

### 除錯模式

使用 `--test-mode` 檢視詳細資訊:
- 原始模型回應
- 選擇提取結果
- API呼叫詳情
- 頻率計算

## 範例

### 簡單權威偏差測試
```bash
python api_bias.py --bias authority --model gpt4 --samples 5 --test-mode
```

### 全面多模型比較
```bash
# 跨多個模型測試權威偏差
for model in gpt4 claude gemini; do
    python api_bias.py --bias authority --model $model --samples 15
done
```

### 自訂偏差級別分析
```bash
python api_bias.py --bias framing --levels "0,20,40,60,80,100" --samples 20
```
