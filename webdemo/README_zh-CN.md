# CoBRA Web演示

使用Transformers `TextIteratorStreamer` 的最小FastAPI后端 + 静态前端,实现真实token流式传输。

## 设置

创建虚拟环境(推荐)并安装依赖:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r webdemo/requirements.txt
# 为您的平台安装torch。仅CPU版本:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

可选(用于受限模型):

```bash
export HF_TOKEN=your_token
```

## 运行

```bash
python -m uvicorn webdemo.backend.main:app --host 0.0.0.0 --port 8010 --reload
```

打开UI:

- http://localhost:8010/ui/

## 注意事项
- 默认模型: `NousResearch/Hermes-3-Llama-3.1-8B`。在UI或后端 `DEFAULT_MODEL` 中更改。
- 滑块0-1映射到Likert 0-4,可选死区(参见 `/calibration/{bias}`)。
- 流式传输是服务器发送事件(SSE);前端在token到达时追加。
