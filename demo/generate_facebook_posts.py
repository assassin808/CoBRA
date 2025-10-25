"""Generate Facebook-style posts (positive & negative) using multiple OpenRouter models.

Extended features:
 - Original default: 100 POSITIVE + 100 NEGATIVE (use --total N).
 - New: independent --pos-total / --neg-total (override --total per class).
 - New: --only-positive / --only-negative selective regeneration (do not touch other file unless missing).
 - Designed to allow scaling one pool (e.g. negatives to 1000) for better statistical power.

Updates:
 - Separate generation passes for POSITIVE and NEGATIVE posts (better control & balance).
 - Multi‑model diversity: qwen, gpt4.1, claude, gemini, grok (configurable).
 - Each post requested at ~128 tokens (rich detail, natural tone).
 - Parser now accepts optional numbering (e.g., '1. POSITIVE: ...').
 - Exact cap: 100 positive / 100 negative (or --total override) – truncate excess.
 - No silent fallbacks; any API failure raises.

Accepted Line Formats:
 POSITIVE: text ...
 12. POSITIVE: text ...
 NEGATIVE: text ...
 7. NEGATIVE: text ...

Example (numbered form accepted):
POSITIVE POSTS:
1. POSITIVE: I just wrapped a long run and feel incredibly energized by the cool morning air.
2. POSITIVE: Our community garden harvest was huge today; sharing baskets with neighbors feels amazing.
...

NEGATIVE POSTS:
1. NEGATIVE: The commute drained me; sitting in traffic for an hour ruins the rest of the evening.
2. NEGATIVE: Another project delay and zero acknowledgment from management—really demoralizing.
...

Output files:
    demo/data/facebook_positive_posts.json (skipped if --only-negative)
    demo/data/facebook_negative_posts.json (skipped if --only-positive)
"""
from __future__ import annotations
import os, json, re, time, argparse, math, sys
from typing import List, Tuple, Dict
import requests

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

POS_FILE = os.path.join(OUTPUT_DIR, 'facebook_positive_posts.json')
NEG_FILE = os.path.join(OUTPUT_DIR, 'facebook_negative_posts.json')

POS_PROMPT_TEMPLATE = (
    "Generate {n} distinct POSITIVE Facebook user posts, each ~128 tokens (rich detail, coherent, natural).\n"
    "STRICT FORMAT ONLY.\n\n"
    "POSITIVE POSTS:\n"
    "1. POSITIVE: <text>\n"
    "2. POSITIVE: <text>\n"
    "(Continue numbering until exactly {n} lines)\n\n"
    "Constraints:\n"
    "- Each line starts with optional numbering + 'POSITIVE:' token.\n"
    "- No extra commentary before or after sections. Only the lines.\n"
)

NEG_PROMPT_TEMPLATE = (
    "Generate {n} distinct NEGATIVE Facebook user posts, each ~128 tokens (authentic, nuanced, not abusive).\n"
    "STRICT FORMAT ONLY.\n\n"
    "NEGATIVE POSTS:\n"
    "1. NEGATIVE: <text>\n"
    "2. NEGATIVE: <text>\n"
    "(Continue numbering until exactly {n} lines)\n\n"
    "Constraints:\n"
    "- Each line starts with optional numbering + 'NEGATIVE:' token.\n"
    "- No extra commentary before or after sections. Only the lines.\n"
)

MODEL_NAME_MAP: Dict[str, str] = {
    'qwen': 'qwen/qwen3-30b-a3b',
    'gpt4.1': 'openai/gpt-4.1',
    'gpt4': 'openai/gpt-4.1',  # alias
    'claude': 'anthropic/claude-sonnet-4',
    'gemini': 'google/gemini-2.5-pro',
    'grok': 'x-ai/grok-4',
}


def call_openrouter(api_key: str, prompt: str, model: str, max_tokens: int) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a data generator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2)

NUMBERED_RE = re.compile(r'^\s*(\d+)[).]\s*(POSITIVE:|NEGATIVE:)\s*(.*)$', re.IGNORECASE)

def parse_posts(text: str) -> Tuple[List[str], List[str]]:
    positive, negative = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = NUMBERED_RE.match(line)
        if m:
            tag = m.group(2).upper()
            content = m.group(3).strip()
            if tag.startswith('POSITIVE:') and content:
                positive.append(content)
            elif tag.startswith('NEGATIVE:') and content:
                negative.append(content)
            continue
        if line.upper().startswith('POSITIVE:'):
            content = line[len('POSITIVE:'):].strip()
            if content:
                positive.append(content)
        elif line.upper().startswith('NEGATIVE:'):
            content = line[len('NEGATIVE:'):].strip()
            if content:
                negative.append(content)
    return positive, negative

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openrouter-key', default=os.getenv('OPENROUTER_API_KEY'), required=False)
    ap.add_argument('--models', nargs='*', default=['qwen','gpt4.1','claude','gemini','grok'], help='Model aliases to use (see MODEL_NAME_MAP)')
    ap.add_argument('--force', action='store_true', help='Regenerate even if files exist (applies only to sides not skipped)')
    ap.add_argument('--total', type=int, default=100, help='Legacy: target count for BOTH positive & negative (each).')
    ap.add_argument('--pos-total', type=int, default=None, help='Target POSITIVE count (overrides --total for positives)')
    ap.add_argument('--neg-total', type=int, default=None, help='Target NEGATIVE count (overrides --total for negatives)')
    ap.add_argument('--only-positive', action='store_true', help='Generate ONLY positive posts (leave existing negatives untouched)')
    ap.add_argument('--only-negative', action='store_true', help='Generate ONLY negative posts (leave existing positives untouched)')
    ap.add_argument('--max-tokens-per-call', type=int, default=40960, help='OpenRouter max_tokens per request')
    ap.add_argument('--sleep', type=float, default=2.0, help='Seconds between calls')
    args = ap.parse_args()

    if not args.openrouter_key:
        print("ERROR: Provide OpenRouter key via --openrouter-key or OPENROUTER_API_KEY env var")
        sys.exit(1)

    if args.only_positive and args.only_negative:
        print("ERROR: Cannot specify both --only-positive and --only-negative.")
        sys.exit(1)

    target_pos = args.pos_total if args.pos_total is not None else args.total
    target_neg = args.neg_total if args.neg_total is not None else args.total

    pos_exists = os.path.exists(POS_FILE)
    neg_exists = os.path.exists(NEG_FILE)
    need_pos = (not args.only_negative) and (args.force or not pos_exists)
    need_neg = (not args.only_positive) and (args.force or not neg_exists)
    if not need_pos and not need_neg:
        print("Nothing to do (files exist). Use --force or selective flags to regenerate.")
        return

    models = []
    for m in args.models:
        if m.lower() in MODEL_NAME_MAP:
            models.append(m.lower())
        else:
            print(f"WARNING: Unknown model alias '{m}' – skipping.")
    if not models:
        raise RuntimeError("No valid models provided.")

    def generate_pool(tag: str, target: int, template: str) -> List[str]:
        per_model = math.ceil(target / len(models))
        acc: List[str] = []
        print(f"=== Generating {tag} posts (target={target}) ===")
        for idx, alias in enumerate(models, 1):
            if len(acc) >= target:
                break
            need = target - len(acc)
            req = min(per_model, need)
            prompt = template.format(n=req)
            model_name = MODEL_NAME_MAP[alias]
            print(f"[{tag[:3]} {idx}] {alias}: requesting {req}")
            txt = call_openrouter(args.openrouter_key, prompt, model_name, args.max_tokens_per_call)
            pos, neg = parse_posts(txt)
            lines = pos if tag == 'POSITIVE' else neg
            print(f"  returned {len(lines)} {tag} lines")
            acc.extend(lines[:req])
            time.sleep(args.sleep)
        acc = acc[:target]
        if len(acc) < target:
            raise RuntimeError(f"Did not reach required {tag} count: {len(acc)}/{target}")
        return acc

    # Preserve untouched side if skipping
    if not need_pos and pos_exists:
        with open(POS_FILE) as f:
            all_pos = json.load(f)
        print(f"[Skip] Keeping existing POSITIVE ({len(all_pos)})")
    else:
        all_pos = generate_pool('POSITIVE', target_pos, POS_PROMPT_TEMPLATE) if need_pos else []
    if not need_neg and neg_exists:
        with open(NEG_FILE) as f:
            all_neg = json.load(f)
        print(f"[Skip] Keeping existing NEGATIVE ({len(all_neg)})")
    else:
        all_neg = generate_pool('NEGATIVE', target_neg, NEG_PROMPT_TEMPLATE) if need_neg else []

    if need_pos:
        with open(POS_FILE, 'w') as f:
            json.dump(all_pos, f, ensure_ascii=False, indent=2)
        print("Saved:", POS_FILE)
    if need_neg:
        with open(NEG_FILE, 'w') as f:
            json.dump(all_neg, f, ensure_ascii=False, indent=2)
        print("Saved:", NEG_FILE)
    print(f"Final counts -> POSITIVE: {len(all_pos)} NEGATIVE: {len(all_neg)}")

if __name__ == '__main__':
    main()
