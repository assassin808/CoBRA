# CoBRA Web示範

使用Transformers `TextIteratorStreamer` 的最小FastAPI後端 + 靜態前端,實現真實token串流傳輸。

## 設定

建立虛擬環境(推薦)並安裝相依套件:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r webdemo/requirements.txt
# 為您的平台安裝torch。僅CPU版本:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

可選(用於受限模型):

```bash
export HF_TOKEN=your_token
```

## 執行

```bash
python -m uvicorn webdemo.backend.main:app --host 0.0.0.0 --port 8010 --reload
```

開啟UI:

- http://localhost:8010/ui/

## 注意事項
- 預設模型: `NousResearch/Hermes-3-Llama-3.1-8B`。在UI或後端 `DEFAULT_MODEL` 中變更。
- 滑桿0-1對映到Likert 0-4,可選死區(參見 `/calibration/{bias}`)。
- 串流傳輸是伺服器傳送事件(SSE);前端在token到達時追加。
