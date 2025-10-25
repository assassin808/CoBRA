from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid as _uuid
import uvicorn
import json
import os
import sys
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Iterator
from threading import Thread
import glob
import random
import hashlib
import time
import threading
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from transformers.generation.streamers import TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from datetime import datetime

app = FastAPI(title="CoBRA Web Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
SUPPORTED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "NousResearch/Hermes-3-Llama-3.1-8B",
]
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Respect offline mode to avoid network (fail fast if not cached)
HF_OFFLINE = str(os.getenv("HF_HUB_OFFLINE", "0")).lower() in ("1", "true", "yes")

# Debug flag: enable verbose logging when environment sets debug=1 (or DEBUG=1)
DEBUG = str(os.getenv("debug") or os.getenv("DEBUG") or "0").lower() in ("1", "true", "yes", "y", "on")

# Calibration logging controls
CALIB_LOG_ENABLED = str(os.getenv("CALIB_LOG") or "0").lower() in ("1", "true", "yes", "on") or DEBUG
CALIB_LOG_TOPK = str(os.getenv("CALIB_LOG_TOPK") or "1").lower() in ("1", "true", "yes", "on")
CALIB_LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(CALIB_LOG_DIR, exist_ok=True)
CALIB_LOG_PATH = os.path.join(CALIB_LOG_DIR, "calibration.log")
CALIB_MAX_POINTS = int(os.getenv("CALIB_MAX_POINTS") or 21)

def _calib_log(event: str, **fields):
    if not CALIB_LOG_ENABLED:
        return
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        **fields,
    }
    try:
        with open(CALIB_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        try:
            print(f"[WARN] calibration logging failed: {e}")
        except Exception:
            pass

def _topk_from_probs(prob_vector: torch.Tensor, tok: AutoTokenizer, k: int = 10) -> list[dict]:
    try:
        vals, idxs = torch.topk(prob_vector, k)
        vals = vals.detach().cpu().tolist()
        idxs = idxs.detach().cpu().tolist()
        out = []
        for p, i in zip(vals, idxs):
            try:
                token = tok.decode([int(i)], skip_special_tokens=False)
            except Exception:
                token = str(tok.convert_ids_to_tokens(int(i)))
            out.append({"id": int(i), "token": token, "prob": float(p)})
        return out
    except Exception as e:
        print(f"[WARN] topk extraction failed: {e}")
        return []

def dbg(msg: str, **fields):
    if not DEBUG:
        return
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        def _fmt(v):
            if isinstance(v, (int, float, bool)) or v is None:
                return v
            if isinstance(v, str):
                if len(v) <= 400:
                    return v
                return v[:400] + f"... (len {len(v)})"
            return f"<{type(v).__name__}>"
        payload = {k: _fmt(v) for k, v in fields.items()}
        print(f"[DEBUG] {ts} {msg} :: {json.dumps(payload, ensure_ascii=False)}")
    except Exception:
        print(f"[DEBUG] {ts} {msg} :: {fields}")

def _safe_torch_load(path: str, map_location=None):
    """Load a PyTorch file robustly across versions.
    Explicitly set weights_only=False (PyTorch >=2.6 safety default) and
    gracefully fall back for older versions that don't support the flag.
    """
    try:
        # Newer PyTorch supports weights_only; ensure full object dict load
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only parameter
        return torch.load(path, map_location=map_location)

BIAS_DESCRIPTIONS = {
    "authority": "Tendency to comply with authority figures.",
    "bandwagon": "Bandwagon effect: adopting beliefs because many others do.",
    "framing": "Choices influenced by how options are presented.",
    "confirmation": "Favoring information that confirms preconceptions.",
}

# Minimal per-bias calibration schema
CALIBRATION = {
    # slider [0,1] -> Likert [0,4], with dead zones (unadjustable) per bias
    "bandwagon": {
        "likert_levels": [0, 1, 2, 3, 4],
        "dead_zones": [],
        "mapping": "linear",  # linear piecewise outside dead zones
        "note": "Demo-only mapping; replace with real CBI curve."
    },
    "authority": {
        "likert_levels": [0, 1, 2, 3, 4],
        "dead_zones": [],
        "mapping": "linear"
    },
    "framing": {"likert_levels": [0,1,2,3,4], "dead_zones": [], "mapping": "linear"},
    "confirmation": {"likert_levels": [0,1,2,3,4], "dead_zones": [], "mapping": "linear"}
}

# Serve frontend at /ui and redirect / to /ui
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

# Project root and data dir
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Register custom RepE pipelines if available
try:
    from control.repe.pipelines import repe_pipeline_registry
    repe_pipeline_registry()
except Exception as e:
    # Make availability explicit so you know what's running
    print(f"[WARN] RepE pipeline registry unavailable: {e}. RepE features may be partially or fully disabled.")

# Import RepE helpers and env config
try:
    from control.repe_experiment import (
        get_controlled_choice_probabilities,
        find_dynamic_control_coeffs,
    )
    from control.experiment_utils import evaluate_repread_accuracy
    from control.env_config import BATCH_SIZE, TEMPERATURE, STEP_SIZE
except Exception as e:
    print(f"[WARN] RepE helpers/env not available: {e}. Falling back to limited functionality.")
    get_controlled_choice_probabilities = None
    find_dynamic_control_coeffs = None
    evaluate_repread_accuracy = None
    BATCH_SIZE = 1
    TEMPERATURE = 1.0
    STEP_SIZE = 1.0

# Dataset loader for bias scenarios/templates
try:
    # Allow namespace import for examples.unified_bias
    EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples", "unified_bias")
    if EXAMPLES_DIR not in sys.path:
        sys.path.append(EXAMPLES_DIR)
    from utils_bias import BiasDataManager, load_scenarios_and_options
except Exception as e:
    print(f"[WARN] Dataset loader unavailable: {e}. Using fallback scenario text only.")
    BiasDataManager = None
    load_scenarios_and_options = None

# Ephemeral in-memory cache for calibration curves (no disk persistence)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
_CACHE_LOCK = threading.Lock()
_RUNTIME_CALIB: Dict[str, Any] = {}
DISK_CALIB_DIR = os.path.join(CACHE_DIR, "calibrations")
os.makedirs(DISK_CALIB_DIR, exist_ok=True)

# Simple in-process RepE contexts per (model,bias)
REPE_CTX: Dict[str, Dict[str, Any]] = {}
REPE_DIR = os.path.join(os.path.dirname(__file__), "cache", "rep_directions")
os.makedirs(REPE_DIR, exist_ok=True)

# Lightweight in-memory progress tracker for long-running curve builds
_PROGRESS_LOCK = threading.Lock()
_PROGRESS: Dict[str, Dict[str, Any]] = {}

# Per-request cancellation registry for streaming generation
_CANCEL_LOCK = threading.Lock()
_CANCEL_EVENTS: Dict[str, threading.Event] = {}

def _register_cancel_event(request_id: str) -> threading.Event:
    with _CANCEL_LOCK:
        ev = _CANCEL_EVENTS.get(request_id)
        if ev is None:
            ev = threading.Event()
            _CANCEL_EVENTS[request_id] = ev
        return ev

def _pop_cancel_event(request_id: str) -> None:
    with _CANCEL_LOCK:
        if request_id in _CANCEL_EVENTS:
            _CANCEL_EVENTS.pop(request_id, None)

def _progress_set(key: str, **fields):
    try:
        with _PROGRESS_LOCK:
            rec = _PROGRESS.get(key, {})
            rec.update(fields)
            rec["ts"] = time.time()
            _PROGRESS[key] = rec
        if DEBUG:
            dbg("progress.update", key=key, **fields)
    except Exception as e:
        # Surface failures instead of swallowing
        try:
            print(f"[WARN] progress.update failed for key={key}: {e}")
        except Exception:
            # Last-resort: avoid crashing
            sys.stderr.write(f"[WARN] progress.update failed for key={key}: {e}\n")

def _safe_model_id(mid: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in (mid or ''))

def _role_fingerprint(role: str) -> str:
    return hashlib.sha1((role or '').encode('utf-8')).hexdigest()[:16]

def _rep_dirs_path(model_id: str, bias: str) -> str:
    return os.path.join(REPE_DIR, f"{_safe_model_id(model_id)}__bias_{(bias or '').lower()}.pt")


def _model_tags(model_id: str):
    mid = (model_id or "").lower()
    if "qwen" in mid:
        user_tag = "<|im_start|>user\n"
        assistant_tag = "<|im_end|>\n<|im_start|>assistant/no_think\n<think>\n</think>\n"
        assistant_prompt_for_choice = "Answer: "
    elif "mistral" in mid:
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
        assistant_prompt_for_choice = " Answer: "
    else:
        user_tag = "USER: "
        assistant_tag = "ASSISTANT: "
        assistant_prompt_for_choice = "Answer: "
    # thinking tag optional in our usage
    thinking_assistant_tag = assistant_tag
    return user_tag, assistant_tag, assistant_prompt_for_choice, thinking_assistant_tag


def _load_cache() -> Dict[str, Any]:
    # Return the in-memory calibration records only (disk is loaded at startup)
    return dict(_RUNTIME_CALIB)


def _save_cache(cache: Dict[str, Any]) -> None:
    # Update the in-memory cache only; on-disk persistence handled at call sites
    _RUNTIME_CALIB.clear()
    _RUNTIME_CALIB.update(cache)


def _calib_file_path(key: str) -> str:
    return os.path.join(DISK_CALIB_DIR, f"{key}.json")


def _persist_record(key: str, record: Dict[str, Any]) -> None:
    try:
        path = _calib_file_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        if DEBUG:
            dbg("calib.disk_saved", key=key, path=os.path.relpath(path, PROJECT_ROOT))
    except Exception as e:
        if DEBUG:
            dbg("calib.disk_save_failed", key=key, error=str(e))


def _init_disk_cache() -> None:
    # One-time load of any persisted calibration records into memory
    try:
        files = sorted(glob.glob(os.path.join(DISK_CALIB_DIR, "*.json")))
        merged: Dict[str, Any] = {}
        for jf in files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            meta = rec.get("meta") or {}
            try:
                new_key = _calib_key(meta)
            except Exception:
                # Fall back to filename key if key computation fails
                new_key = os.path.splitext(os.path.basename(jf))[0]
            # Prefer lower-resolution (fewer points) when merging duplicates
            def npoints(r: Dict[str, Any]) -> int:
                xs = r.get("x")
                return len(xs) if isinstance(xs, list) else 0
            if new_key not in merged:
                merged[new_key] = rec
            else:
                if npoints(rec) < npoints(merged[new_key]):
                    merged[new_key] = rec
        _RUNTIME_CALIB.clear()
        _RUNTIME_CALIB.update(merged)
        if DEBUG:
            dbg("calib.disk_loaded", count=len(merged), dir=os.path.relpath(DISK_CALIB_DIR, PROJECT_ROOT))
    except Exception as e:
        if DEBUG:
            dbg("calib.disk_load_failed", error=str(e))


# Load disk cache at import time
_init_disk_cache()


def _calib_key(payload: Dict[str, Any]) -> str:
    # Stable hash over selected fields
    method = (payload.get("control_method") or "prompt").lower()
    if method == "repe":
        # Calibration validity depends on model, bias, role, method and operator
        include_keys = ["bias", "model", "role_prompt", "control_method", "repe_operator", "temperature", "experiment_name"]
    else:
        # Prompt curve is dataset-averaged; include temperature to match bias.py behavior
        include_keys = [
            "bias", "model", "role_prompt",
            "control_method", "experiment_name", "temperature"
        ]
    data = {k: payload.get(k) for k in include_keys}
    s = json.dumps(data, sort_keys=True, ensure_ascii=False)
    key = hashlib.sha1(s.encode("utf-8")).hexdigest()
    dbg("calib_key", method=method, include_keys=include_keys, key=key)
    return key


# Map UI-visible method labels to internal method/operator
def _normalize_method(control_method: str | None, repe_operator: str | None) -> tuple[str, str | None, str]:
    m = (control_method or "prompt").strip()
    op = repe_operator
    label = "Prompt_numerical"
    ml = m.lower()
    if ml in ("prompt", "prompt_numerical", "prompt-numerical"):
        return "prompt", None, "Prompt_numerical"
    if ml in ("repe linear", "repe_linear", "repe-linear", "repe"):
        return "repe", "linear_comb", "RepE Linear"
    if ml in ("repe projection", "repe_projection", "repe-projection"):
        return "repe", "projection", "RepE Projection"
    # Fallback to existing fields
    if (repe_operator or "").lower() in ("projection",):
        return "repe", "projection", "RepE Projection"
    if (repe_operator or "").lower() in ("linear_comb", "linear-comb", "linear"):
        return "repe", "linear_comb", "RepE Linear"
    return "prompt", None, "Prompt_numerical"

@app.get("/")
async def root_redirect():
    if os.path.isdir(FRONTEND_DIR):
        return RedirectResponse(url="/ui/")
    return {"message": "CoBRA Web Demo backend running. Frontend not found."}

class GenerateRequest(BaseModel):
    bias: str
    model: str = DEFAULT_MODEL
    role_prompt: str = ""
    scenario: str
    bias_level: float  # legacy control; kept for backward-compat (0..1 or coeff)
    target_cbi: float | None = None  # desired CBI (0..4); preferred over bias_level when provided
    temperature: float = 0.7
    max_new_tokens: int = 256
    control_method: str = "prompt"  # accepts: "prompt"/"Prompt_numerical"/"RepE Linear"/"RepE Projection"
    repe_operator: str | None = None  # "linear_comb" | "projection"
    experiment_name: str | None = None  # optional dataset experiment for prompt curves (e.g., 'asch','hotel')
    request_id: str | None = None  # SSE cancel token


def get_model(model_id: str):
    if model_id in MODEL_CACHE:
        dbg("get_model.cache_hit", model=model_id)
        return MODEL_CACHE[model_id]["tok"], MODEL_CACHE[model_id]["mdl"]
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left", legacy=False, local_files_only=HF_OFFLINE)
        # Use float16 when CUDA available; else float32 for CPU
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=HF_OFFLINE,
        )
    except Exception as e:
        hint = " Set HF_HUB_OFFLINE=1 and pre-download the model if you are offline."
        raise HTTPException(500, f"Failed to load model/tokenizer '{model_id}': {e}.{hint}")
    # Ensure pad token exists
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    MODEL_CACHE[model_id] = {"tok": tok, "mdl": mdl}
    dbg("get_model.loaded", model=model_id, hf_offline=HF_OFFLINE, dtype=str(mdl.dtype) if hasattr(mdl, "dtype") else None)
    return tok, mdl


def slider_to_likert(bias: str, level: float) -> float:
    """Continuous mapping: slider [0,1] -> Likert [0,4] (float). Dead zones ignored for continuity."""
    if level < 0: level = 0.0
    if level > 1: level = 1.0
    return float(level * 4.0)


def _build_choice_token_map(tok: AutoTokenizer) -> Dict[str, list]:
    """Target token map for A–E including numeric variants, mirroring base.py.
    Includes single-token variants for letters and numbers with/without leading whitespace.
    """
    labels = ["A", "B", "C", "D", "E"]
    mapping: Dict[str, list] = {}
    for i, l in enumerate(labels):
        number_label = str(i + 1)
        str_variants = [
            l, f" {l}", f"\t{l}",
            number_label, f" {number_label}", f"\t{number_label}",
            f"{l}.", f" {l}.", f" ({l})", f" {l})", f" {l} )",
        ]
        ids = set()
        for s in str_variants:
            try:
                enc = tok.encode(s, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            except Exception:
                continue
        mapping[l] = list(ids)
    return mapping


def _compute_next_token_probs(mdl: AutoModelForCausalLM, tok: AutoTokenizer, prompt_text: str, temperature: float | None = None) -> torch.Tensor:
    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mdl(**inputs)
    logits = outputs.logits[:, -1, :].squeeze(0)
    if temperature is not None and float(temperature) != 1.0:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    return probs


def _compute_cbi_from_probs(prob_vector: torch.Tensor, token_map: Dict[str, list]) -> Dict[str, float]:
    choice_probs: Dict[str, float] = {}
    for label, ids in token_map.items():
        if not ids:
            choice_probs[label] = 0.0
        else:
            idx = torch.tensor(ids, device=prob_vector.device, dtype=torch.long)
            choice_probs[label] = float(prob_vector[idx].sum().item())
    # Normalize for stable score; if total is zero, return neutral 2.0 like base.py
    total = sum(choice_probs.values())
    if total > 0:
        norm = {k: v / total for k, v in choice_probs.items()}
        weights = {l: (4 - i) for i, l in enumerate(["A", "B", "C", "D", "E"])}
        cbi = sum(norm[l] * weights[l] for l in weights)
    else:
        norm = {k: 0.0 for k in choice_probs}
        cbi = 2.0
    return {"cbi": cbi, "total": total, "normalized": norm, "raw": choice_probs}


def _ensure_repe_trained(bias: str, model_id: str, tok: AutoTokenizer, mdl: AutoModelForCausalLM, role_prompt: str | None = None, train_if_missing: bool = True) -> Dict[str, Any]:
    """Ensure a trained RepReader + RepControl pipeline exist for (model,bias)."""
    # Use model+role as the persistence key; bias variations reuse same directions as requested
    # Note: we still namespace REPE_CTX by model+bias to avoid cross-bias collisions in-memory
    key = f"{model_id}::{bias.lower()}"
    ctx = REPE_CTX.get(key)
    if ctx:
        dbg("repe.ctx_hit", model=model_id, bias=bias,
            layers=len(ctx.get("control_layers", [])))
        return ctx

    if BiasDataManager is None or get_controlled_choice_probabilities is None:
        raise HTTPException(500, "RepE prerequisites not available: BiasDataManager or RepE helpers missing.")

    # Try loading persisted directions for the current (model, bias)
    user_tag, assistant_tag, assistant_prompt_for_choice, thinking_tag = _model_tags(model_id)
    # New scheme: persist per model+bias (role-agnostic)
    dirs_path = _rep_dirs_path(model_id, bias)
    try:
        dbg("repe.load_dirs_check", path=dirs_path, exists=os.path.exists(dirs_path))
    except Exception as e:
        print(f"[WARN] Debug log failed (repe.load_dirs_check): {e}")
    try:
        if os.path.exists(dirs_path):
            blob = _safe_torch_load(dirs_path, map_location=mdl.device)
            directions = blob.get("directions")
            direction_signs = blob.get("direction_signs")
            control_layers = blob.get("control_layers")
            if directions and direction_signs:
                dbg("repe.load_dirs", path=dirs_path, scheme="model+bias",
                    layers=len(directions),
                    layer_ids=sorted(list(directions.keys()))[:15])
                class _SimpleReader:
                    pass
                rep_reader = _SimpleReader()
                rep_reader.directions = directions
                rep_reader.direction_signs = direction_signs
                if not control_layers:
                    control_layers = sorted(directions.keys())
                rep_control_pipeline = hf_pipeline("rep-control", model=mdl, tokenizer=tok, layers=control_layers, control_method="reading_vec")
                dbg("repe.ctx_init", control_layers=control_layers[:15], layers_total=len(control_layers))
                # Console visibility for demo logs
                try:
                    print(f"[INFO] RepE control layers (loaded): {control_layers[:15]}{'...' if len(control_layers)>10 else ''}")
                except Exception as e:
                    print(f"[WARN] Failed to print loaded control layers: {e}")
                ctx = {
                    "rep_control_pipeline": rep_control_pipeline,
                    "rep_reader": rep_reader,
                    "control_layers": control_layers,
                    "user_tag": user_tag,
                    "assistant_tag": assistant_tag,
                    "assistant_prompt_for_choice": assistant_prompt_for_choice,
                    "thinking_assistant_tag": thinking_tag,
                }
                REPE_CTX[key] = ctx
                return ctx
    except Exception as e:
        # If loading fails, fall back to training path
        dbg("repe.load_dirs_failed", path=dirs_path, error=str(e))

    # Legacy fallback: load older role-fingerprinted file specific to this bias+role+tags
    try:
        legacy_fingerprint = f"{bias.lower()}|{role_prompt or ''}|{user_tag}|{assistant_tag}"
        legacy_path = os.path.join(REPE_DIR, f"{_safe_model_id(model_id)}__{_role_fingerprint(legacy_fingerprint)}.pt")
        if os.path.exists(legacy_path):
            blob = _safe_torch_load(legacy_path, map_location=mdl.device)
            directions = blob.get("directions")
            direction_signs = blob.get("direction_signs")
            control_layers = blob.get("control_layers")
            if directions and direction_signs:
                dbg("repe.load_dirs_legacy", path=legacy_path, scheme="model+role+bias",
                    layers=len(directions), layer_sample=sorted(list(directions.keys()))[:15])
                class _SimpleReader:
                    pass
                rep_reader = _SimpleReader()
                rep_reader.directions = directions
                rep_reader.direction_signs = direction_signs
                if not control_layers:
                    control_layers = sorted(directions.keys())
                rep_control_pipeline = hf_pipeline("rep-control", model=mdl, tokenizer=tok, layers=control_layers, control_method="reading_vec")
                try:
                    print(f"[INFO] RepE control layers (loaded-legacy): {control_layers[:15]}{'...' if len(control_layers)>10 else ''}")
                except Exception as e:
                    print(f"[WARN] Failed to print loaded legacy control layers: {e}")
                ctx = {
                    "rep_control_pipeline": rep_control_pipeline,
                    "rep_reader": rep_reader,
                    "control_layers": control_layers,
                    "user_tag": user_tag,
                    "assistant_tag": assistant_tag,
                    "assistant_prompt_for_choice": assistant_prompt_for_choice,
                    "thinking_assistant_tag": thinking_tag,
                }
                REPE_CTX[key] = ctx
                return ctx
    except Exception as e:
        dbg("repe.load_dirs_legacy_failed", model=model_id, bias=bias, error=str(e))

    # Train reader directions
    try:
        if not train_if_missing:
            raise HTTPException(400, "RepReader directions not found. Please run Calibrate (RepE) once to train directions.")
        data_manager = BiasDataManager(EXAMPLES_DIR)
        dataset = data_manager.create_training_dataset(bias.lower(), tok, user_tag, assistant_tag, testing=False, model_name=None)
        if not dataset or not dataset.get("train", {}).get("data"):
            raise RuntimeError("Training dataset unavailable for RepReader")
        dbg("repe.training_start", model=model_id, bias=bias,
            train_size=len(dataset['train']['data']) if dataset.get('train') else 0,
            val_size=len(dataset.get('val', {}).get('data', [])) if isinstance(dataset.get('val'), dict) else 0,
            test_size=len(dataset.get('test', {}).get('data', [])) if isinstance(dataset.get('test'), dict) else 0,
            dataset_split="train")

        rep_token = -1
        hidden_layers = list(range(-1, -mdl.config.num_hidden_layers - 1, -1))
        rep_reading = hf_pipeline("rep-reading", model=mdl, tokenizer=tok)
        rep_reader = rep_reading.get_directions(
            dataset['train']['data'],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            train_labels=dataset['train']['labels'],
            direction_method='pca',
            n_difference=1,
            batch_size=BATCH_SIZE,
            direction_finder_kwargs={'n_components': 4},
        )
        dbg("repe.training_directions_done",
            layers=len(rep_reader.directions) if getattr(rep_reader, 'directions', None) else 0,
            sample_layers=sorted(list(rep_reader.directions.keys()))[:15])
        # Evaluate and pick top control layers
        results, best_layer = evaluate_repread_accuracy(
            rep_reading, dataset, rep_token, hidden_layers, rep_reader,
            plot_dir=os.path.join(os.path.dirname(__file__), "cache"), debug=DEBUG
        ) if evaluate_repread_accuracy else ({}, None)
        dbg("repe.training_eval_done", best_layer=best_layer if best_layer is not None else -1,
            results_count=len(results) if isinstance(results, dict) else 0)
        if rep_reader and getattr(rep_reader, 'directions', None):
            control_layers = sorted(rep_reader.directions.keys(), key=lambda l: (results.get(l, 0.0) if isinstance(results, dict) else 0.0), reverse=True)[:15]
        else:
            raise RuntimeError("RepReader produced no directions.")

        rep_control_pipeline = hf_pipeline("rep-control", model=mdl, tokenizer=tok, layers=control_layers, control_method="reading_vec")
        dbg("repe.control_layers_selected", top_k=len(control_layers), layer_ids=control_layers[:15])
        try:
            print(f"[INFO] RepE control layers (selected): {control_layers[:15]}{'...' if len(control_layers)>10 else ''}")
        except Exception as e:
            print(f"[WARN] Failed to print selected control layers: {e}")

        ctx = {
            "rep_control_pipeline": rep_control_pipeline,
            "rep_reader": rep_reader,
            "control_layers": control_layers,
            "user_tag": user_tag,
            "assistant_tag": assistant_tag,
            "assistant_prompt_for_choice": assistant_prompt_for_choice,
            "thinking_assistant_tag": thinking_tag,
        }
        # Persist directions for future reuse keyed by (model, bias) only
        try:
            blob = {
                "directions": rep_reader.directions,
                "direction_signs": rep_reader.direction_signs,
                "control_layers": control_layers,
                "model_id": model_id,
                "role_tags": {"user": user_tag, "assistant": assistant_tag},
            }
            torch.save(blob, dirs_path)
            dbg("repe.training_saved", path=dirs_path, scheme="model+bias",
                layers=len(control_layers))
        except Exception:
            dbg("repe.training_save_failed", path=dirs_path)
        REPE_CTX[key] = ctx
        return ctx
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"RepReader training failed: {e}")


def _compute_next_token_probs_repe(tok: AutoTokenizer, mdl: AutoModelForCausalLM, prompt_text: str, ctx: Dict[str, Any], coeff: float, operator: str = "linear_comb") -> torch.Tensor:
    rep_ctrl = ctx["rep_control_pipeline"]
    rep_reader = ctx.get("rep_reader")
    control_layers = ctx["control_layers"]
    device = mdl.device
    model_dtype = next(rep_ctrl.wrapped_model.parameters()).dtype

    if rep_reader is None:
        raise HTTPException(500, "RepReader not trained for this (model,bias). Calibrate first.")
    directions = rep_reader.directions
    direction_signs = rep_reader.direction_signs
    activations = {
        layer: torch.tensor(
            coeff * float(direction_signs[layer][0]) * directions[layer][0]
        ).to(device).to(model_dtype)
        for layer in control_layers
    }

    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    rep_ctrl.wrapped_model.reset()
    rep_ctrl.wrapped_model.set_controller(control_layers, activations, block_name="decoder_block", operator=operator or "linear_comb")
    with torch.no_grad():
        outputs = rep_ctrl.wrapped_model.model(**inputs)
    rep_ctrl.wrapped_model.reset()
    logits = outputs.logits[:, -1, :].squeeze(0)
    return torch.softmax(logits, dim=-1)


def _build_chat_for_method(
    req: GenerateRequest,
    likert_cont: float,
    *,
    include_mcq: bool = False,
    scenario_override: str | None = None,
) -> list[dict]:
    # Normalize control method so 'Prompt_numerical' is treated as prompt
    method_norm, _op_norm, _label = _normalize_method(req.control_method, req.repe_operator)
    if method_norm == "prompt":
        pct = round(likert_cont / 4.0 * 100)
        bias_type = (req.bias or "authority").lower()
        desc_map = {
            "authority": {
                "max": "always trust authority figures",
                "min": "never trust authority figures",
            },
            "bandwagon": {
                "max": "always follow what the majority does",
                "min": "never follow what the majority does",
            },
            "framing": {
                "max": "always be strongly influenced by how information is presented",
                "min": "never be influenced by how information is presented",
            },
            "confirmation": {
                "max": "always seek only information that confirms your existing beliefs",
                "min": "never seek only information that confirms your existing beliefs",
            },
        }
        d = desc_map.get(bias_type, desc_map["authority"])
        ctrl = (
            f"Pretend you have {bias_type} bias at level {pct}% out of 100% "
            f"(where 0%={d['min']}, 100%={d['max']}). "
            f"Each level increases bias by exactly 5%. "
            f"Act with this precise level {pct}% of {bias_type} bias."
        )
    else:
        # For RepE we do not rely on prompt text for control; keep system role only
        ctrl = None

    # System content: only role (omit if empty). Control text is in user content for prompt method.
    sys_prompt = f"Role: {req.role_prompt}" if (req.role_prompt or "").strip() else ""

    scenario_text = scenario_override if scenario_override is not None else req.scenario
    # If natural generation, strip any Options block from scenario
    if not include_mcq and isinstance(scenario_text, str) and 'Options:' in scenario_text:
        scenario_text = scenario_text.split('Options:')[0].rstrip()
    # Build user content. For prompt control, put control at the very top.
    if method_norm == "prompt" and ctrl:
        user_content = f"{ctrl}\n\n{scenario_text}"
    else:
        user_content = f"{scenario_text}"
    if include_mcq and isinstance(scenario_text, str) and 'Choose one of the options' not in scenario_text:
        user_content += "\nChoose one of the options (A, B, C, D, E) directly, e.g., 'Answer: X'."

    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user_content})
    if include_mcq:
        # Bias.py places the Answer: on assistant side to probe next-token prob
        _u, _a, assistant_prompt_for_choice, _t = _model_tags(req.model)
        msgs.append({"role": "assistant", "content": assistant_prompt_for_choice})
    # For RepE we don't apply prompt-based control; avoid misleading likert=0.0 in logs
    _likert_for_log = likert_cont if method_norm == "prompt" else None
    dbg("build_chat", method=method_norm, include_mcq=include_mcq, likert=_likert_for_log,
        scenario_len=len(scenario_text) if isinstance(scenario_text, str) else None)
    return msgs


def _interp1(xs: List[float], ys: List[float], x: float) -> float:
    """Linear interpolate/extrapolate y at x over sorted xs."""
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    # Clamp
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    # Find interval
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, y0 = xs[i-1], ys[i-1]
            x1, y1 = xs[i], ys[i]
            if x1 == x0:
                return float(y0)
            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(ys[-1])


def _derive_control_range(xs: List[float], ys: List[float]) -> List[float]:
    if not xs or not ys or len(xs) != len(ys):
        return [0.0, 1.0]
    y_min = min(ys)
    y_max = max(ys)
    span = y_max - y_min
    if span <= 1e-6:
        return [0.0, 1.0]
    eps = 0.05
    y_low = y_min + eps * span
    y_high = y_min + (1 - eps) * span
    cand_low = [x for x, y in zip(xs, ys) if y >= y_low]
    cand_high = [x for x, y in zip(xs, ys) if y >= y_high]
    if not cand_low or not cand_high:
        return [0.0, 1.0]
    lo = float(min(cand_low))
    hi = float(max(cand_high))
    if not (lo < hi):
        return [0.0, 1.0]
    return [round(lo, 6), round(hi, 6)]


def _inverse_xs_for_y(xs: List[float], ys: List[float], y_target: float) -> List[float]:
    """Find all x where linearly interpolated y(x) == y_target across piecewise segments.
    Returns possibly multiple x values. If no crossings, returns empty list.
    """
    if not xs or not ys or len(xs) != len(ys):
        return []
    sol: List[float] = []
    for i in range(1, len(xs)):
        x0, y0 = xs[i-1], ys[i-1]
        x1, y1 = xs[i], ys[i]
        dy0 = y0 - y_target
        dy1 = y1 - y_target
        # Crossing or touching within segment
        if dy0 == 0.0:
            sol.append(float(x0))
        if dy0 == 0.0 and dy1 == 0.0:
            # Flat segment equal to target: include mid-point
            sol.append(float((x0 + x1) / 2.0))
        elif dy0 * dy1 <= 0.0 and (x1 != x0) and (y1 != y0):
            t = (y_target - y0) / (y1 - y0)
            if 0.0 <= t <= 1.0:
                sol.append(float(x0 + t * (x1 - x0)))
    # Deduplicate with tolerance
    sol_sorted = sorted(sol)
    out: List[float] = []
    for v in sol_sorted:
        if not out or abs(out[-1] - v) > 1e-6:
            out.append(v)
    return out


def _pick_nearest_to_zero(values: List[float]) -> float | None:
    if not values:
        return None
    return sorted(values, key=lambda v: abs(v))[0]


def _nearest_x_for_y(xs: List[float], ys: List[float], y_target: float) -> float | None:
    """Find x on piecewise-linear curve closest in y to target. Returns best x or None."""
    if not xs or not ys or len(xs) != len(ys):
        return None
    best_x = xs[0]
    best_err = abs(ys[0] - y_target)
    for i in range(1, len(xs)):
        x0, y0 = xs[i-1], ys[i-1]
        x1, y1 = xs[i], ys[i]
        dy = y1 - y0
        dx = x1 - x0
        if dy == 0:
            cand_x = 0.5 * (x0 + x1)
            cand_y = y0
        else:
            t = (y_target - y0) / dy
            if t < 0:
                cand_x = x0; cand_y = y0
            elif t > 1:
                cand_x = x1; cand_y = y1
            else:
                cand_x = x0 + t * dx
                # linear interpolation
                cand_y = y0 + t * dy
        err = abs(cand_y - y_target)
        if err < best_err:
            best_err = err
            best_x = cand_x
    return float(best_x)


def _compute_prompt_curve_points(req: GenerateRequest, tok: AutoTokenizer, mdl: AutoModelForCausalLM, num_points: int = 21, progress_key: str | None = None) -> tuple[List[float], List[float]]:
    """Compute prompt-based CBI curve points (xs in [0,1], ys in [0,4]) using dataset-averaged MCQ if available."""
    token_map = _build_choice_token_map(tok)
    xs = [i / (num_points - 1) for i in range(max(2, num_points))]
    ys: List[float] = []
    scenarios = None; mcq_options = None; prompt_template = None
    if load_scenarios_and_options is not None:
        try:
            scenarios, mcq_options, prompt_template = load_scenarios_and_options(req.bias, experiment_name=req.experiment_name)
        except Exception:
            scenarios = None
    dbg("prompt_curve.init", num_points=num_points, experiment=req.experiment_name,
        scenarios=len(scenarios) if scenarios else 0,
        dataset_source="utils_bias" if (scenarios and prompt_template) else "none",
        dataset_split="eval")
    # Initialize progress if requested
    if progress_key:
        try:
            if scenarios and mcq_options and prompt_template:
                total_samples = len(xs) * len(scenarios)
            else:
                total_samples = len(xs)
            _progress_set(progress_key, status="running", samples_total=total_samples, samples_done=0, points_total=len(xs), points_done=0)
        except Exception as e:
            print(f"[WARN] Failed to init prompt curve progress: {e}")
    for i_pt, x in enumerate(xs, start=1):
        ctrl = float(max(0.0, min(1.0, x)))
        likert_cont = slider_to_likert(req.bias, ctrl)
        if scenarios and mcq_options and prompt_template:
            opt_keys = sorted(list(mcq_options.keys()))
            options_block = "\n".join([f"{k}: {mcq_options[k]}" for k in opt_keys])
            subset = (scenarios or [])  # Use all scenarios like bias.py
            cbi_vals: List[float] = []
            for i_s, s in enumerate(subset, start=1):
                try:
                    base_prompt = prompt_template.format(**s)
                except Exception:
                    base_prompt = s.get("Statement") or s.get("statement") or str(s)
                combined = (
                    f"{base_prompt}\nOptions:\n{options_block}\n"
                    "Choose one of the options (A, B, C, D, E) directly, e.g., 'Answer: X'."
                )
                chat = _build_chat_for_method(req, likert_cont, include_mcq=True, scenario_override=combined)
                prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                probs = _compute_next_token_probs(mdl, tok, prompt_text, temperature=req.temperature)
                if CALIB_LOG_ENABLED and CALIB_LOG_TOPK:
                    _calib_log(
                        "prompt_curve_sample",
                        bias=req.bias,
                        model=req.model,
                        method="prompt",
                        temperature=req.temperature,
                        experiment=req.experiment_name,
                        point_index=i_pt,
                        points_total=len(xs),
                        scenario_index=i_s,
                        x=ctrl,
                        likert=likert_cont,
                        question=str(base_prompt)[:400],
                        prompt_tail=str(prompt_text)[-400:],
                        topk=_topk_from_probs(probs, tok, k=10),
                    )
                out = _compute_cbi_from_probs(probs, token_map)
                cbi_vals.append(out["cbi"])
                if DEBUG and (i_s % 10 == 0 or i_s == len(subset)):
                    dbg("prompt_curve.sample_progress", point=i_pt, points=len(xs), sample=i_s, samples=len(subset))
                if progress_key:
                    _progress_set(progress_key, samples_done=((i_pt-1)*len(subset) + i_s), points_done=(i_pt if i_s == len(subset) else i_pt-1))
            avg = float(sum(cbi_vals)/len(cbi_vals)) if cbi_vals else 0.0
            ys.append(avg)
            dbg("prompt_curve.point", x=ctrl, y=avg, samples=len(cbi_vals),
                dataset="all_scenarios_avg", options=len(opt_keys))
        else:
            chat = _build_chat_for_method(req, likert_cont, include_mcq=True)
            prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            probs = _compute_next_token_probs(mdl, tok, prompt_text, temperature=req.temperature)
            if CALIB_LOG_ENABLED and CALIB_LOG_TOPK:
                _calib_log(
                    "prompt_curve_point",
                    bias=req.bias,
                    model=req.model,
                    method="prompt",
                    temperature=req.temperature,
                    experiment=req.experiment_name,
                    point_index=i_pt,
                    points_total=len(xs),
                    x=ctrl,
                    likert=likert_cont,
                    question=None,
                    prompt_tail=str(prompt_text)[-400:],
                    topk=_topk_from_probs(probs, tok, k=10),
                )
            out = _compute_cbi_from_probs(probs, token_map)
            ys.append(out["cbi"])
            if progress_key:
                _progress_set(progress_key, samples_done=i_pt, points_done=i_pt)
    dbg("prompt_curve.done", points=len(xs))
    if progress_key:
        _progress_set(progress_key, status="done")
    return xs, ys


def stream_generate(req: GenerateRequest) -> Iterator[bytes]:
    """True token-by-token streaming using TextIteratorStreamer.
    For RepE method, applies the controller during generation. No MCQ wrapping; clean free-form output.
    """
    try:
        tok, mdl = get_model(req.model)
        # Ensure request id and register cancel event
        request_id = req.request_id or str(_uuid.uuid4())
        cancel_event = _register_cancel_event(request_id)
        method_norm, operator_norm, method_label = _normalize_method(req.control_method, req.repe_operator)
        dbg("generate.start", method=method_norm, operator=operator_norm, target_cbi=req.target_cbi, bias=req.bias, request_id=request_id)
        # Determine control value (0..1 for prompt, coefficient for RepE) if target CBI is provided
        control_cont = float(max(0.0, min(1.0, req.bias_level)))
        if method_norm != "repe" and isinstance(req.target_cbi, (int, float)):
            payload = {
                "bias": req.bias,
                "model": req.model,
                "role_prompt": req.role_prompt,
                "control_method": method_norm,
                "repe_operator": operator_norm,
                "experiment_name": req.experiment_name,
        "temperature": req.temperature,
            }
            key = _calib_key(payload)
            with _CACHE_LOCK:
                recp = _load_cache().get(key)
            xs = ys = None
            if recp and recp.get("x") and recp.get("y"):
                dbg("generate.prompt_curve_cache_hit", key=key, points=len(recp.get("x") or []))
                xs = recp["x"]; ys = recp["y"]
            else:
                # Compute prompt curve on the fly (dataset-averaged if available) with 21 points like bias.py
                prog_key = f"{req.bias}:{req.experiment_name or 'default'}:prompt"
                xs, ys = _compute_prompt_curve_points(req, tok, mdl, num_points=CALIB_MAX_POINTS, progress_key=prog_key)
                # Save it in memory so subsequent calls reuse it
                record = {
                    "x": xs,
                    "x_plot": xs,
                    "y": ys,
                    "range": [0.0, 1.0],
                    "meta": payload,
                    "ts": time.time(),
                    "version": 1,
                    "method": method_norm,
                    "method_label": method_label,
                }
                with _CACHE_LOCK:
                    cache = _load_cache()
                    cache[key] = record
                    _save_cache(cache)
                # Persist to disk as well for demo speed
                _persist_record(key, record)
            # Invert to all solutions then pick the one nearest to 0 (no-control preference)
            sol_xs = _inverse_xs_for_y(xs, ys, float(req.target_cbi))
            picked = _pick_nearest_to_zero(sol_xs) if sol_xs else None
            if picked is None:
                # Pick x whose y is nearest to target
                picked = _nearest_x_for_y(xs, ys, float(req.target_cbi))
            if picked is not None:
                control_cont = float(max(0.0, min(1.0, picked)))
            dbg("generate.prompt_inversion", target=req.target_cbi, picked=control_cont)

        # Before building prompt, emit resolved control for UI display
        try:
            if method_norm != "repe":
                yield f"data: {json.dumps({'event':'control','method':method_norm,'control':control_cont,'target_cbi': req.target_cbi, 'request_id': request_id})}\n\n".encode()
            # RepE branch emits after coefficient is resolved below
        except Exception as e:
            print(f"[WARN] SSE control preface failed: {e}")
        # Build prompt after choosing control
        likert = slider_to_likert(req.bias, control_cont)
        tmp_req = GenerateRequest(**{**req.model_dump(), "bias_level": control_cont})
        # Natural free-form generation: do not include MCQ tail
        chat = _build_chat_for_method(tmp_req, likert, include_mcq=False)
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        class EventStoppingCriteria(StoppingCriteria):
            def __init__(self, ev: threading.Event):
                self.ev = ev
            def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
                return self.ev.is_set()
        stopping_criteria = StoppingCriteriaList([EventStoppingCriteria(cancel_event)])

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=0.95,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=stopping_criteria,
        )
        if method_norm == "repe":
            # Require prior calibration to map slider->coef
            payload = {
                "bias": req.bias,
                "model": req.model,
                "role_prompt": req.role_prompt,
                "control_method": method_norm,
                "repe_operator": operator_norm,
                "temperature": req.temperature,
                "experiment_name": req.experiment_name,
            }
            key = _calib_key(payload)
            # Safe read without lock: _load_cache returns a copy
            rec = _load_cache().get(key)
            # Back-compat: if not found with operator-aware key, try legacy key without operator
            if (not rec) or ("coef_min" not in rec or "coef_max" not in rec):
                legacy_payload = {**payload}
                legacy_payload.pop("repe_operator", None)
                legacy_key = _calib_key(legacy_payload)
                rec_legacy = _load_cache().get(legacy_key)
                if rec_legacy:
                    rec = rec_legacy
            if not rec or "coef_min" not in rec or "coef_max" not in rec:
                err = {"error": f"{method_label} generation requires prior calibration. Please run Calibrate first."}
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            coef_min = rec["coef_min"]; coef_max = rec["coef_max"]
            # If Target CBI provided and curve present, invert to pick coeff near zero
            coeff = None
            if isinstance(req.target_cbi, (int, float)) and rec.get("x") and rec.get("y"):
                sol_xs = _inverse_xs_for_y(rec["x"], rec["y"], float(req.target_cbi))
                picked = _pick_nearest_to_zero(sol_xs) if sol_xs else None
                if picked is not None:
                    coeff = float(min(max(picked, coef_min), coef_max))
            if coeff is None:
                control_cont = float(req.bias_level)
                # If value not within coef range, assume 0..1 slider and map
                if control_cont < coef_min or control_cont > coef_max:
                    control_cont = float(max(0.0, min(1.0, control_cont)))
                    coeff = float(coef_min + (coef_max - coef_min) * control_cont)
                else:
                    coeff = float(control_cont)
            dbg("generate.repe_coeff", coef_min=coef_min, coef_max=coef_max, coeff=coeff)
            # Emit resolved coefficient for UI display (no range or resolved CBI)
            try:
                yield f"data: {json.dumps({'event':'control','method':'repe','coefficient':coeff,'target_cbi': req.target_cbi, 'request_id': request_id})}\n\n".encode()
            except Exception as e:
                print(f"[WARN] SSE RepE control preface failed: {e}")

            dbg("generate.repe.ensure_dirs", bias=req.bias, model=req.model)
            ctx = _ensure_repe_trained(req.bias, req.model, tok, mdl, role_prompt=req.role_prompt, train_if_missing=False)
            dbg("generate.repe.have_dirs", ok=bool(ctx))
            rep_ctrl = ctx["rep_control_pipeline"]
            rep_reader = ctx.get("rep_reader")
            control_layers = ctx["control_layers"]
            model_dtype = next(rep_ctrl.wrapped_model.parameters()).dtype
            activations = {
                layer: torch.tensor(
                    coeff * float(rep_reader.direction_signs[layer][0]) * rep_reader.directions[layer][0]
                ).to(mdl.device).to(model_dtype)
                for layer in control_layers
            }

            rep_ctrl.wrapped_model.reset()
            rep_ctrl.wrapped_model.set_controller(
                control_layers, activations, block_name="decoder_block", operator=(operator_norm or "linear_comb")
            )

            thread = Thread(target=rep_ctrl.wrapped_model.model.generate, kwargs=gen_kwargs)
            thread.start()
            cancelled = False
            for new_text in streamer:
                if cancel_event.is_set():
                    cancelled = True
                    break
                yield f"data: {json.dumps({'token': new_text})}\n\n".encode()

            thread.join()
            rep_ctrl.wrapped_model.reset()
            if cancelled:
                try:
                    yield f"data: {json.dumps({'event':'cancelled','request_id': request_id})}\n\n".encode()
                except Exception:
                    pass
            yield b"data: [DONE]\n\n"
            dbg("generate.done", cancelled=cancelled)
        else:
            thread = Thread(target=mdl.generate, kwargs=gen_kwargs)
            thread.start()
            cancelled = False
            for new_text in streamer:
                if cancel_event.is_set():
                    cancelled = True
                    break
                yield f"data: {json.dumps({'token': new_text})}\n\n".encode()

            thread.join(timeout=0.1)
            if cancelled:
                try:
                    yield f"data: {json.dumps({'event':'cancelled','request_id': request_id})}\n\n".encode()
                except Exception:
                    pass
            yield b"data: [DONE]\n\n"
            dbg("generate.done", cancelled=cancelled)
    except Exception as e:
        err = {"error": str(e)}
        yield f"data: {json.dumps(err)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        try:
            if 'request_id' in locals():
                _pop_cancel_event(request_id)
        except Exception:
            pass


@app.get("/models")
async def list_models():
    return {"default": DEFAULT_MODEL, "available": SUPPORTED_MODELS}


@app.get("/biases")
async def list_biases():
    return {"biases": list(BIAS_DESCRIPTIONS.keys()), "descriptions": BIAS_DESCRIPTIONS}


@app.get("/experiments/{bias}")
async def list_experiments(bias: str):
    if BiasDataManager is None:
        return {"experiments": []}
    try:
        data_manager = BiasDataManager(EXAMPLES_DIR)
        names = data_manager.get_experiment_names(bias)
        return {"experiments": names or []}
    except Exception as e:
        print(f"[WARN] list_experiments failed: {e}")
        return {"experiments": []}


@app.get("/calibration/{bias}")
async def get_calibration(bias: str):
    conf = CALIBRATION.get(bias)
    if not conf:
        raise HTTPException(404, "Unknown bias")
    return conf


@app.get("/default_scenario/{bias}")
async def default_scenario(bias: str, experiment_name: str | None = None):
    bias = bias.lower()
    # Prefer utils_bias loader so we format the true scenario using prompt_template
    try:
        if load_scenarios_and_options is not None:
            scenarios, mcq_options, prompt_template = load_scenarios_and_options(bias, experiment_name=experiment_name)
            if scenarios and prompt_template:
                s = scenarios[0]
                try:
                    text = prompt_template.format(**s)
                except Exception:
                    text = s.get("Statement") or s.get("statement") or str(s)
                return {"bias": bias, "scenario": text, "experiment_name": experiment_name}
    except Exception as e:
        print(f"[WARN] default_scenario utils_bias path failed: {e}")
    # Fallback to raw data files if utils_bias not available
    bias_dir = os.path.join(DATA_DIR, bias)
    files = sorted(glob.glob(os.path.join(bias_dir, "*.json")))
    random.shuffle(files)
    for jf in files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                random.shuffle(data)
                for rec in data:
                    if isinstance(rec, dict) and isinstance(rec.get("Statement"), str):
                        return {"bias": bias, "scenario": rec["Statement"].strip(), "source_file": os.path.relpath(jf, PROJECT_ROOT)}
        except Exception as e:
            print(f"[WARN] default_scenario file parse failed for {jf}: {e}")
            continue
    return {"bias": bias, "scenario": "You see several posts about a topic. What do you do next?"}


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest):
    return StreamingResponse(stream_generate(req), media_type="text/event-stream")


@app.post("/cancel/{request_id}")
async def cancel_generation(request_id: str):
    with _CANCEL_LOCK:
        ev = _CANCEL_EVENTS.get(request_id)
    if not ev:
        return {"status": "not-found", "request_id": request_id}
    try:
        ev.set()
        return {"status": "cancelling", "request_id": request_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to cancel: {e}")


@app.post("/cbi_preview")
async def cbi_preview(req: GenerateRequest):
    """Compute per-scenario CBI as in base.py: expected Likert from A–E choice probabilities.
    This uses a generic MCQ tail so it doesn't depend on original datasets.
    """
    tok, mdl = get_model(req.model)
    method_norm, operator_norm, method_label = _normalize_method(req.control_method, req.repe_operator)
    # Back-compat: if users pass old fields, method_norm/operator_norm reflect it.
    control_cont = float(max(0.0, min(1.0, req.bias_level)))
    likert_cont = slider_to_likert(req.bias, control_cont)

    # Fast path: if calibration cached, interpolate y without recomputing logits
    payload = {
        "bias": req.bias,
        "model": req.model,
        "role_prompt": req.role_prompt,
        "control_method": method_norm,
        "repe_operator": operator_norm,
        "experiment_name": req.experiment_name,
        "temperature": req.temperature,
    }
    key = _calib_key(payload)
    with _CACHE_LOCK:
        rec = _load_cache().get(key)
    dbg("preview.lookup", key=key, cached=bool(rec))
    if DEBUG:
        try:
            rp = (req.role_prompt or "").strip()
            print(f"[INFO] Preview role prompt loaded: {'<empty>' if not rp else (rp[:80] + ('...' if len(rp)>80 else ''))}")
        except Exception:
            pass
    # Back-compat for RepE: try legacy key without operator
    if method_norm == "repe" and not rec:
        legacy_payload = {**payload}
        legacy_payload.pop("repe_operator", None)
        legacy_key = _calib_key(legacy_payload)
        with _CACHE_LOCK:
            rec = _load_cache().get(legacy_key)

    if rec and isinstance(rec.get("x"), list) and isinstance(rec.get("y"), list):
        xs = rec["x"]; ys = rec["y"]
        # If target_cbi provided, invert y->x and choose coefficient closest to 0 (no-control)
        if isinstance(req.target_cbi, (int, float)):
            sol_xs = _inverse_xs_for_y(xs, ys, float(req.target_cbi))
            picked = _pick_nearest_to_zero(sol_xs) if sol_xs else None
            if picked is not None:
                cbi_val = float(req.target_cbi)
                return {
                    "bias": req.bias,
                    "control": picked,
                    "likert_control": slider_to_likert(req.bias, 0.0),
                    "choice_probs": {},
                    "normalized": {},
                    "cbi": cbi_val,
                    "cbi01": cbi_val/4.0,
                    "cached": True,
                    "method_label": method_label,
                }
        # Otherwise use provided control/coeff
        if "coef_min" in rec and "coef_max" in rec and len(xs) == len(ys):
            coeff = control_cont
            if coeff < rec["coef_min"] or coeff > rec["coef_max"]:
                coeff = rec["coef_min"] + (rec["coef_max"] - rec["coef_min"]) * control_cont
            cbi_val = _interp1(xs, ys, float(coeff))
            chosen_control = float(coeff)
        else:
            cbi_val = _interp1(xs, ys, control_cont)
            chosen_control = float(control_cont)
        resp = {
            "bias": req.bias,
            "control": chosen_control,
            "likert_control": likert_cont,
            "choice_probs": {},
            "normalized": {},
            "cbi": cbi_val,
            "cbi01": cbi_val/4.0,
            "cached": True,
            "method_label": method_label,
        }
        dbg("preview.cached", control=chosen_control, cbi=cbi_val)
        return resp

    # Fallback: compute once (no caching here)
    dbg("preview.compute_fallback", dataset_source="none", include_mcq=True)
    chat = _build_chat_for_method(req, likert_cont, include_mcq=True)
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    method = method_norm
    if method == "repe":
        # Require calibration to map slider to coeff; if missing, train directions and then continue
        if not rec or "coef_min" not in rec or "coef_max" not in rec:
            # Train directions to avoid hard failure; user might have calibration from disk but not loaded context
            try:
                _ensure_repe_trained(req.bias, req.model, tok, mdl, role_prompt=req.role_prompt, train_if_missing=True)
            except Exception as e:
                raise HTTPException(400, f"RepE preview requires prior calibration (run Calibrate first). Details: {e}")
            # Still cannot map coefficient range without calibration curve; return helpful error
            raise HTTPException(400, "RepE preview needs calibration curve to map slider to coefficient. Please run Calibrate.")
        coef_min = rec["coef_min"]; coef_max = rec["coef_max"]
        # If target CBI provided, invert curve to pick coefficient closest to zero
        if isinstance(req.target_cbi, (int, float)) and rec.get("x") and rec.get("y"):
            sol_xs = _inverse_xs_for_y(rec["x"], rec["y"], float(req.target_cbi))
            picked = _pick_nearest_to_zero(sol_xs) if sol_xs else None
            if picked is not None:
                coeff = float(min(max(picked, coef_min), coef_max))
            else:
                # fallback to mapping control
                coeff = float(coef_min + (coef_max - coef_min) * control_cont)
        else:
            # Interpret req.bias_level as coefficient when it sits in [coef_min,coef_max]; else map 0..1 -> [coef_min,coef_max]
            coeff = float(control_cont)
            if coeff < coef_min or coeff > coef_max:
                coeff = float(coef_min + (coef_max - coef_min) * control_cont)
        dbg("preview.repe.ensure_dirs", bias=req.bias, model=req.model)
        ctx = _ensure_repe_trained(req.bias, req.model, tok, mdl, role_prompt=req.role_prompt, train_if_missing=False)
        if not ctx:
            # Auto-train directions if missing so preview can proceed
            dbg("preview.repe.train_missing", action="train_if_missing")
            ctx = _ensure_repe_trained(req.bias, req.model, tok, mdl, role_prompt=req.role_prompt, train_if_missing=True)
        dbg("preview.repe.have_dirs", ok=bool(ctx))
        prob_vec = _compute_next_token_probs_repe(
            tok, mdl, prompt_text, ctx, coeff=coeff, operator=(operator_norm or "linear_comb")
        )
    else:
        prob_vec = _compute_next_token_probs(mdl, tok, prompt_text, temperature=req.temperature)
    token_map = _build_choice_token_map(tok)
    out = _compute_cbi_from_probs(prob_vec, token_map)
    resp = {
        "bias": req.bias,
        "control": control_cont,
        "likert_control": likert_cont,
        "choice_probs": out["raw"],
        "normalized": out["normalized"],
        "cbi": out["cbi"],
        "cbi01": out["cbi"]/4.0,
        "cached": False,
        "method_label": method_label,
    }
    dbg("preview.computed", control=control_cont, cbi=out["cbi"])
    return resp


@app.post("/cbi_curve")
async def cbi_curve(req: GenerateRequest, num_points: int = 21):
    method_norm, operator_norm, method_label = _normalize_method(req.control_method, req.repe_operator)
    # Clamp requested points to configured maximum (defaults to 21)
    try:
        num_points = max(2, min(int(num_points), int(CALIB_MAX_POINTS)))
    except Exception:
        num_points = 21
    dbg("curve.start", method=method_norm, operator=operator_norm, num_points=num_points, experiment=req.experiment_name)
    if DEBUG:
        try:
            rp = (req.role_prompt or "").strip()
            print(f"[INFO] Calibration role prompt loaded: {'<empty>' if not rp else (rp[:80] + ('...' if len(rp)>80 else ''))}")
        except Exception:
            pass
    prog_key = f"{req.bias}:{req.experiment_name or 'default'}:{method_norm}"
    # Try cache first
    payload = {
        "bias": req.bias,
        "model": req.model,
        "role_prompt": req.role_prompt,
        "control_method": method_norm,
        "repe_operator": operator_norm,
    # num_points is intentionally ignored by cache key to avoid duplicates
    "num_points": num_points,
        "experiment_name": req.experiment_name,
        "temperature": req.temperature,
    }
    key = _calib_key(payload)
    with _CACHE_LOCK:
        cache = _load_cache()
        if key in cache:
            rec = cache[key]
            resp = {"x": rec.get("x"), "x_plot": rec.get("x_plot"), "y": rec.get("y"), "range": rec.get("range"), "x_label": rec.get("x_label"), "cached": True, "key": key, "method_label": rec.get("method_label")}
            if "coef_min" in rec: resp["coef_min"] = rec["coef_min"]
            if "coef_max" in rec: resp["coef_max"] = rec["coef_max"]
            if "coef_domain" in rec: resp["coef_domain"] = rec["coef_domain"]
            dbg("curve.cache_hit", key=key)
            # If this is a RepE curve, ensure directions are trained so preview/generation won't fail
            if method_norm == "repe":
                try:
                    tok, mdl = get_model(req.model)
                    await asyncio.to_thread(_ensure_repe_trained, req.bias, req.model, tok, mdl, req.role_prompt)
                except Exception as e:
                    if DEBUG:
                        dbg("curve.cache_hit.repe_train_failed", error=str(e))
            return resp

    # Compute fresh
    tok, mdl = get_model(req.model)
    token_map = _build_choice_token_map(tok)
    is_repe = method_norm == "repe"
    if not is_repe:
        # Offload prompt curve computation to a worker thread to keep /progress responsive
        xs, ys = await asyncio.to_thread(_compute_prompt_curve_points, req, tok, mdl, num_points, prog_key)
        _calib_log("prompt_curve_done", bias=req.bias, model=req.model, method="prompt", points=len(xs), experiment=req.experiment_name, temperature=req.temperature)
        # For prompt, plotting domain is [0,1]
        record = {
            "x": xs,
            "x_plot": xs,
            "y": ys,
            "range": [0.0, 1.0],
            "x_label": "Control Coefficient",
            "meta": payload,
            "ts": time.time(),
            "version": 1,
            "method": method_norm,
            "method_label": method_label,
        }
        dbg("curve.prompt.done", points=len(xs))
        _progress_set(prog_key, status="done")
    else:
        # Ensure trained RepE context (offload to thread; can be long)
        _progress_set(prog_key, status="running", phase="repe_init")
        ctx = await asyncio.to_thread(_ensure_repe_trained, req.bias, req.model, tok, mdl, req.role_prompt)
        rep_ctrl = ctx["rep_control_pipeline"]
        rep_reader = ctx.get("rep_reader")
        control_layers = ctx["control_layers"]
        user_tag = ctx["user_tag"]; assistant_tag = ctx["assistant_tag"]; assistant_prompt_for_choice = ctx["assistant_prompt_for_choice"]

        # Load scenarios/options for this bias for adaptive sampling
        if load_scenarios_and_options is None:
            raise HTTPException(500, "Missing scenarios loader: utils_bias.load_scenarios_and_options not available.")
        try:
            scenarios, mcq_options, prompt_template = load_scenarios_and_options(req.bias, experiment_name=req.experiment_name)
        except Exception as e:
            raise HTTPException(500, f"Failed to load scenarios/options for bias '{req.bias}': {e}")
        dbg("curve.repe.dataset", source="utils_bias", scenarios=len(scenarios) if scenarios else 0,
            options=len(mcq_options) if mcq_options else 0, experiment=req.experiment_name)

        # MCQ instruction: reuse base 'choice_first' style
        mcq_prompt_addon = "Choose one of the options (A, B, C, D, E) directly, e.g., 'Answer: X'."
        operator = (operator_norm or "linear_comb")

        # Determine coefficient domain via adaptive sampling (offload to thread)
        _progress_set(prog_key, status="running", phase="coeff_search")
        if find_dynamic_control_coeffs is not None and rep_reader is not None:
            coeffs = await asyncio.to_thread(
                find_dynamic_control_coeffs,
                rep_ctrl, tok, rep_reader, control_layers,
                token_map, mdl.device,
                scenarios or [], mcq_options, prompt_template,
                mcq_prompt_addon, rep_reader.direction_signs, rep_reader.directions,
                user_tag, assistant_tag, assistant_tag, assistant_prompt_for_choice,
                "choice_first", operator,
                req.temperature or TEMPERATURE,
                0.9, 100, STEP_SIZE, 0.001, DEBUG,
            )
            coeffs = list(map(float, coeffs)) if coeffs is not None and len(coeffs) > 0 else [0.0, 0.5, 1.0]
            dbg("curve.repe.coeff_search", method="adaptive", steps=len(coeffs) if isinstance(coeffs, list) else None)
        else:
            coeffs = [0.0, 0.25, 0.5, 0.75, 1.0]
            dbg("curve.repe.coeff_search", method="fixed", steps=len(coeffs))

        coef_min = min(coeffs); coef_max = max(coeffs)
        # Use raw coefficients as x-axis values for backend logic
        xs = list(map(float, coeffs))
        # Compute normalized x for plotting: map [coef_min, coef_max] -> [0,1]
        denom = (coef_max - coef_min)
        if denom > 0:
            x_plot = [float((v - coef_min) / denom) for v in xs]
        else:
            x_plot = [0.5 for _ in xs]

        # Compute dataset-averaged CBI at these coefficients using multiple scenarios in a worker thread
        def _avg_loop():
            ys_local = []
            opt_keys = sorted(list(mcq_options.keys()))
            options_block = "\n".join([f"{k}: {mcq_options[k]}" for k in opt_keys])
            scenarios_subset = (scenarios or [])
            _progress_set(prog_key, status="running", phase="averaging", coeffs_total=len(coeffs), coeffs_done=0, samples_total=len(scenarios_subset)*len(coeffs), samples_done=0)
            for i_c, c in enumerate(coeffs, start=1):
                cbi_vals = []
                model_dtype = next(rep_ctrl.wrapped_model.parameters()).dtype
                activations = {
                    layer: torch.tensor(
                        c * float(rep_reader.direction_signs[layer][0]) * rep_reader.directions[layer][0]
                    ).to(mdl.device).to(model_dtype)
                    for layer in control_layers
                }
                for i_s, s in enumerate(scenarios_subset, start=1):
                    try:
                        base_prompt = prompt_template.format(**s)
                    except Exception:
                        base_prompt = s.get("Statement") or s.get("statement") or str(s)
                    combined = (
                        f"{base_prompt}\nOptions:\n{options_block}\n"
                        "Choose one of the options (A, B, C, D, E) directly, e.g., 'Answer: X'."
                    )
                    chat = _build_chat_for_method(req, slider_to_likert(req.bias, 0.0), include_mcq=True, scenario_override=combined)
                    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                    if get_controlled_choice_probabilities is None:
                        probs = _compute_next_token_probs_repe(tok, mdl, prompt_text, ctx, coeff=c, operator=operator)
                        if CALIB_LOG_ENABLED and CALIB_LOG_TOPK:
                            _calib_log(
                                "repe_curve_sample",
                                bias=req.bias,
                                model=req.model,
                                method="repe",
                                operator=operator,
                                temperature=req.temperature,
                                experiment=req.experiment_name,
                                coeff=float(c),
                                coeff_index=i_c,
                                coeffs_total=len(coeffs),
                                scenario_index=i_s,
                                question=str(base_prompt)[:400],
                                prompt_tail=str(prompt_text)[-400:],
                                topk=_topk_from_probs(probs, tok, k=10),
                            )
                        out = _compute_cbi_from_probs(probs, token_map)
                        cbi_vals.append(out["cbi"])
                    else:
                        choice_probs = get_controlled_choice_probabilities(
                            rep_ctrl, tok, prompt_text, activations, token_map, mdl.device,
                            debug=False, operator=operator, temperature=req.temperature or TEMPERATURE,
                        )
                        if CALIB_LOG_ENABLED and CALIB_LOG_TOPK:
                            try:
                                # Extra forward for top-k only when logging is requested
                                probs_full = _compute_next_token_probs_repe(tok, mdl, prompt_text, ctx, coeff=c, operator=operator)
                                topk = _topk_from_probs(probs_full, tok, k=10)
                            except Exception as e:
                                topk = []
                                print(f"[WARN] RepE topk logging extra pass failed: {e}")
                            _calib_log(
                                "repe_curve_sample",
                                bias=req.bias,
                                model=req.model,
                                method="repe",
                                operator=operator,
                                temperature=req.temperature,
                                experiment=req.experiment_name,
                                coeff=float(c),
                                coeff_index=i_c,
                                coeffs_total=len(coeffs),
                                scenario_index=i_s,
                                question=str(base_prompt)[:400],
                                prompt_tail=str(prompt_text)[-400:],
                                topk=topk,
                            )
                        total = sum(choice_probs.values())
                        if total > 0:
                            norm = {k: v/total for k, v in choice_probs.items()}
                        else:
                            norm = {k: 0.0 for k in token_map.keys()}
                        weights = {l: (4 - i) for i, l in enumerate(["A","B","C","D","E"]) }
                        cbi_vals.append(sum(norm[l]*weights[l] for l in weights))
                    _progress_set(prog_key, samples_done=((i_c-1)*len(scenarios_subset) + i_s), coeffs_done=(i_c if i_s == len(scenarios_subset) else i_c-1))
                ys_local.append(float(sum(cbi_vals) / len(cbi_vals)) if cbi_vals else 0.0)
                dbg("curve.repe.point", coeff=c, y=ys_local[-1], samples=len(cbi_vals), dataset="all_scenarios_avg", coeff_index=i_c, coeff_total=len(coeffs))
            return ys_local

        ys = await asyncio.to_thread(_avg_loop)

        record = {
            "x": xs,              # raw coefficients for backend use (inversion, selection)
            "x_plot": x_plot,      # normalized for frontend plotting (0..1)
            "y": ys,
            "range": [float(coef_min), float(coef_max)],
            "coef_min": float(coef_min),
            "coef_max": float(coef_max),
            "coef_domain": list(map(float, coeffs)),
            "x_label": "Control Coefficient",
            "meta": payload,
            "ts": time.time(),
            "version": 2,
            "method": method_norm,
            "method_label": method_label,
        }
        dbg("curve.repe.done", coeffs=len(xs), coef_min=coef_min, coef_max=coef_max)
        _calib_log("repe_curve_done", bias=req.bias, model=req.model, method="repe", operator=operator, coeffs=len(xs), coef_min=float(coef_min), coef_max=float(coef_max), experiment=req.experiment_name, temperature=req.temperature)
        _progress_set(prog_key, status="done")
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[key] = record
        _save_cache(cache)
    # Persist to disk for reuse across restarts
    _persist_record(key, record)
    dbg("curve.saved", key=key)

    resp = {"x": record["x"], "x_plot": record.get("x_plot"), "y": record["y"], "range": record["range"], "x_label": record.get("x_label"), "cached": False, "key": key, "method_label": record.get("method_label")}
    if "coef_min" in record: resp["coef_min"] = record["coef_min"]
    if "coef_max" in record: resp["coef_max"] = record["coef_max"]
    if "coef_domain" in record: resp["coef_domain"] = record["coef_domain"]
    return resp


@app.get("/calibrations")
async def list_calibrations():
    with _CACHE_LOCK:
        cache = _load_cache()
    dbg("calib.list", count=len(cache))
    items = []
    for key, rec in cache.items():
        meta = rec.get("meta", {})
        items.append({
            "key": key,
            "bias": meta.get("bias"),
            "model": meta.get("model"),
            "method": rec.get("method"),
            "method_label": rec.get("method_label"),
            # repe_operator deprecated: do not expose
            "temperature": meta.get("temperature"),
            "range": rec.get("range"),
            "num_points": meta.get("num_points"),
            "ts": rec.get("ts"),
            "role_prompt": meta.get("role_prompt"),
            "experiment_name": meta.get("experiment_name"),
        })
    # Sort by ts desc
    items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    return {"items": items}

@app.get("/progress")
async def get_progress():
    with _PROGRESS_LOCK:
        # Return shallow copy to avoid mutation races
        snap = {k: dict(v) for k, v in _PROGRESS.items()}
    return {"items": snap}

@app.post("/calibrations/clear")
async def clear_calibrations():
    with _CACHE_LOCK:
        try:
            _RUNTIME_CALIB.clear()
        except Exception as e:
            raise HTTPException(500, f"Failed to clear cache: {e}")
    # Also clear disk cache files
    try:
        for jf in glob.glob(os.path.join(DISK_CALIB_DIR, "*.json")):
            try:
                os.remove(jf)
            except Exception:
                continue
        if DEBUG:
            dbg("calib.disk_cleared", dir=os.path.relpath(DISK_CALIB_DIR, PROJECT_ROOT))
    except Exception as e:
        if DEBUG:
            dbg("calib.disk_clear_failed", error=str(e))
    dbg("calib.cleared")
    return {"ok": True}

class CalibDeleteRequest(BaseModel):
    keys: List[str]

@app.post("/calibrations/delete")
async def delete_calibrations(body: CalibDeleteRequest):
    with _CACHE_LOCK:
        cache = _load_cache()
        for k in (body.keys or []):
            cache.pop(k, None)
        _save_cache(cache)
    # Remove from disk as well
    deleted_files = 0
    for k in (body.keys or []):
        try:
            fp = _calib_file_path(k)
            if os.path.exists(fp):
                os.remove(fp)
                deleted_files += 1
        except Exception:
            continue
    dbg("calib.deleted", deleted=len(body.keys or []))
    if DEBUG:
        dbg("calib.disk_deleted", deleted_files=deleted_files)
    return {"ok": True, "deleted": body.keys}


class CalibGetRequest(BaseModel):
    bias: str
    model: str
    role_prompt: str
    scenario: str
    temperature: float
    control_method: str
    repe_operator: str | None = None
    num_points: int = 21
    experiment_name: str | None = None


@app.post("/calibrations/get")
async def get_calibration_record(body: CalibGetRequest):
    key = _calib_key(body.model_dump())
    with _CACHE_LOCK:
        cache = _load_cache()
        rec = cache.get(key)
    if not rec:
        raise HTTPException(404, "Calibration not found")
    dbg("calib.get", key=key)
    return {"key": key, **rec}


@app.get("/calibrations/by_key/{key}")
async def get_calibration_by_key(key: str):
    with _CACHE_LOCK:
        cache = _load_cache()
        rec = cache.get(key)
    if not rec:
        raise HTTPException(404, "Calibration not found")
    dbg("calib.get_by_key", key=key)
    return {"key": key, **rec}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
