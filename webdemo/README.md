# CoBRA Web Demo

**📖 Language / 语言**: [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

Minimal FastAPI backend + static frontend with real token streaming using Transformers `TextIteratorStreamer`.

## Setup

Create a virtualenv (recommended) and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r webdemo/requirements.txt
# Install torch for your platform. For CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Optional (for gated models):

```bash
export HF_TOKEN=your_token
```

## Run

```bash
python -m uvicorn webdemo.backend.main:app --host 0.0.0.0 --port 8010 --reload
```

Open the UI:

- http://localhost:8010/ui/

## Notes
- Default model: `NousResearch/Hermes-3-Llama-3.1-8B`. Change in UI or backend `DEFAULT_MODEL`.
- Slider 0–1 maps to Likert 0–4 with optional dead zones (see `/calibration/{bias}`).
- Streaming is server-sent events (SSE); frontend appends tokens as they arrive.
