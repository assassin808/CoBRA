"""Bandwagon RepE sentiment sensitivity demo (clean version).

Pipeline:
1. Train RepReader bandwagon direction via existing unified_bias dataset.
2. Dynamically discover control coefficients (linear_comb) with find_dynamic_control_coeffs.
3. Build synthetic feeds from provided Facebook-style positive / negative posts.
4. For each feed size (1..10) and 10 random feeds, generate one new post under each control coefficient using RepE test_time_run.
5. Score sentiment with cardiffnlp model (score = pos_prob - neg_prob) without extra preprocessing.
6. Plot sentiment vs feed size per coefficient.

No fallbacks: any error raises immediately.
"""
import os, json, random, argparse, time, math
import sys
from typing import List, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline
from transformers.utils import logging as hf_logging
try:
    from tqdm import tqdm
except ImportError:  # fallback lightweight tqdm
    def tqdm(it: Iterable, **kwargs):
        return it
from scipy.special import softmax

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UBIAS_DIR = os.path.join(ROOT, 'examples', 'unified_bias')
if ROOT not in sys.path:
    sys.path.append(ROOT)
if UBIAS_DIR not in sys.path:
    sys.path.append(UBIAS_DIR)
try:
    # Register custom RepE pipelines (rep-reading, rep-control)
    from control.repe import repe_pipeline_registry
    repe_pipeline_registry()
except Exception as e:
    raise RuntimeError(f"Failed to register RepE pipelines: {e}")

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

POS_FILE = os.path.join(DATA_DIR, 'facebook_positive_posts.json')
NEG_FILE = os.path.join(DATA_DIR, 'facebook_negative_posts.json')


def load_posts() -> (List[str], List[str]):
    if not (os.path.exists(POS_FILE) and os.path.exists(NEG_FILE)):
        raise FileNotFoundError("Missing facebook_positive_posts.json / facebook_negative_posts.json. Run generate_facebook_posts.py first.")
    with open(POS_FILE) as f: pos = json.load(f)
    with open(NEG_FILE) as f: neg = json.load(f)
    return pos, neg


def build_groups(posts: List[str], seed: int = 42, max_group_size: int = 10, feeds_per_size: int = 10) -> Dict[int, List[List[str]]]:
    """Build sampled feed groups including size 0 (empty feed) up to max_group_size.

    Each entry n -> list of feeds (lists of posts). For n==0 the feeds are empty lists.
    """
    rng = random.Random(seed)
    return {n: [rng.sample(posts, n) if n > 0 else [] for _ in range(feeds_per_size)] for n in range(0, max_group_size + 1)}


def sentiment_model(device: str, fp16: bool = False, compile_model: bool = False):
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else None
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return tok, cfg, model


def sentiment_score(text: str, tok, cfg, model, device: str) -> float:
    enc = tok(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad(): out = model(**enc)
    probs = softmax(out.logits[0].cpu().numpy())
    labels = [cfg.id2label[i].lower() for i in range(len(probs))]
    return float(probs[labels.index('positive')] - probs[labels.index('negative')])


def batch_sentiment_scores(texts: List[str], tok, cfg, model, device: str, max_length: int = 256) -> List[float]:
    if not texts:
        return []
    enc = tok(texts, return_tensors='pt', truncation=True, max_length=max_length, padding=True).to(device)
    with torch.no_grad():
        out = model(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()
    labels = [cfg.id2label[i].lower() for i in range(probs.shape[1])]
    try:
        pos_idx = labels.index('positive'); neg_idx = labels.index('negative')
    except ValueError:
        # Fallback heuristic: assume label order pos / neu / neg or similar
        pos_idx = 2 if probs.shape[1] > 2 else probs.shape[1]-1
        neg_idx = 0
    return [float(p[pos_idx] - p[neg_idx]) for p in probs]


def discover_coeffs(rep_control, tokenizer, rep_reader, control_layers, data_manager, device, debug=False):
    from control.repe_experiment import find_dynamic_control_coeffs
    # Use first bandwagon experiment subset for probing
    exp_names = data_manager.get_experiment_names('bandwagon')
    if not exp_names:
        raise RuntimeError("No bandwagon experiments found for coefficient discovery.")
    first = exp_names[0]
    scenarios = data_manager.load_experiment_scenarios('bandwagon', first, num_scenarios=12, model_name=None)
    mcq_options, prompt_template = data_manager.get_mcq_options_and_templates('bandwagon', first)
    # Minimal RepEControlExperiment instance just to reuse target token map helper
    from control.repe_experiment import RepEControlExperiment
    # Use the underlying base model (rep_control.model) rather than the wrapped_model so that
    # ControlExperiment can access model.config.* attributes (wrapped model lacks .config).
    dummy = RepEControlExperiment(rep_control.model, tokenizer, device, RESULTS_DIR, rep_control, rep_reader, control_layers, is_testing_mode=debug, num_permutations=1, experiment_suffix="_coeff")
    labels = list(mcq_options.keys())
    token_map = dummy._get_target_token_map(labels, {l: str(i+1) for i,l in enumerate(labels)})
    coeffs = find_dynamic_control_coeffs(
        rep_control,
        tokenizer,
        rep_reader,
        control_layers,
        token_map,
        device,
        scenarios=scenarios,
        mcq_options=mcq_options,
        prompt_template=prompt_template,
        mcq_prompt_addon="Choose one option.",
        direction_signs=rep_reader.direction_signs,
        directions=rep_reader.directions,
        user_tag=dummy.user_tag,
        assistant_tag=dummy.assistant_tag,
        thinking_assistant_tag=dummy.assistant_tag,
        assistant_prompt_for_choice=dummy.assistant_prompt_for_choice,
        mcq_scenario_key='choice_first',
        operator='linear_comb',
        choice_temperature=1.0,
        threshold=0.8,
        max_steps=60,
        initial_step=1.0,
        min_step=0.01,
        debug=debug
    )
    return list(coeffs)


def run(args):
    # Allow plot-only mode to run without GPU / model prerequisites.
    if getattr(args, 'plot_only', False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available; requirement is to use GPU first.")
        device = 'cuda'

    raw_gen_path = os.path.join(RESULTS_DIR, 'sentiment_generation_raw.json')
    scored_path = os.path.join(RESULTS_DIR, 'sentiment_generation_scored.json')
    summary_csv = os.path.join(RESULTS_DIR, 'sentiment_summary.csv')

    # Plot-only short circuit: reuse existing scored or summary artifacts.
    if getattr(args, 'plot_only', False):
        import pandas as pd
        if os.path.exists(summary_csv):
            print(f"[Phase][Plot-Only] Loading existing summary: {summary_csv}")
            summary = pd.read_csv(summary_csv)
        elif os.path.exists(scored_path):
            print(f"[Phase][Plot-Only] Building summary from scored file: {scored_path}")
            with open(scored_path) as f:
                scored_records = json.load(f)
            if not scored_records:
                raise RuntimeError("Scored file empty; cannot plot.")
            df = pd.DataFrame(scored_records)
            summary = df.groupby(['pool','group_size','coeff']).sentiment_score.agg(['mean','std','count']).reset_index()
            summary.rename(columns={'mean':'mean_score','std':'std_score','count':'n'}, inplace=True)
        else:
            raise RuntimeError("--plot-only requested but no summary CSV or scored JSON present.")
        # Subset for plot-only: first 4 smallest coefficients (or all if fewer)
        all_coeffs_sorted = sorted(summary['coeff'].unique())
        if len(all_coeffs_sorted) > 4:
            coeffs_for_plot = all_coeffs_sorted[:4]
            print(f"[Phase][Plot-Only] Restricting coefficients for plot to first 4: {coeffs_for_plot}")
        else:
            coeffs_for_plot = all_coeffs_sorted
            print(f"[Phase][Plot-Only] Using all coefficients (<=4): {coeffs_for_plot}")
        plot(summary, coeffs_for_plot)
        if getattr(args, 'negative_custom_plot', False):
            coeff_order = [-0.625, -0.247, 0.0, 0.43359375, 0.736]
            cbi_vals = [2.55, 2.80, 3.00, 3.11, 3.13]
            coeff_map = {c: cbi for c, cbi in zip(coeff_order, cbi_vals)}
            present = sorted(summary['coeff'].unique())
            overlap = [c for c in coeff_order if c in present]
            if not overlap:
                print(f"[CustomPlot] None of the requested coefficients {coeff_order} are present in summary; skipping custom negative plots.")
            else:
                if len(overlap) < len(coeff_order):
                    print(f"[CustomPlot] Partial overlap {overlap}; plotting available subset.")
                # Rebuild map/order with overlap only
                coeff_order_use = overlap
                coeff_map_use = {c: coeff_map[c] for c in overlap}
                import pandas as pd  # ensure alias
                plot_negative_custom(summary, coeff_map_use, coeff_order_use, RESULTS_DIR, comic_sans_ttf=getattr(args, 'comic_sans_ttf', ''))
        return

    # Load facebook post pools
    pos_posts, neg_posts = load_posts()

    # Apply test-mode overrides before building groups
    if args.test_mode:
        # Conservative small defaults unless user already set smaller values
        if args.max_group_size > 3:
            args.max_group_size = 3
        if args.feeds_per_size > 2:
            args.feeds_per_size = 2
        print(f"[Phase] Test mode active: max_group_size={args.max_group_size} feeds_per_size={args.feeds_per_size}")

    pos_groups = build_groups(pos_posts, seed=args.seed, max_group_size=args.max_group_size, feeds_per_size=args.feeds_per_size)
    neg_groups = build_groups(neg_posts, seed=args.seed + 17, max_group_size=args.max_group_size, feeds_per_size=args.feeds_per_size)

    # Load model via unified bias tagging logic
    from control.experiment_utils import load_model_and_tags
    model, tokenizer, user_tag, assistant_tag, assistant_prompt_for_choice, device_model = load_model_and_tags(args.model)
    if device_model != device:
        model.to(device)

    # Optionally quiet HF warnings
    if args.quiet_hf:
        hf_logging.set_verbosity_error()

    # RepReader training / loading using unified bias dataset (bandwagon)
    from utils_bias import BiasDataManager, ensure_plot_dir
    data_manager = BiasDataManager(UBIAS_DIR)
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers - 1, -1))
    rep_reader_cache = os.path.join(RESULTS_DIR, 'rep_reader')
    rep_reader_meta = os.path.join(rep_reader_cache, 'meta.json')
    os.makedirs(rep_reader_cache, exist_ok=True)
    rep_reading = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    # Always build dataset (needed for accuracy plot even if reader cached)
    dataset = data_manager.create_training_dataset('bandwagon', tokenizer, user_tag=user_tag, assistant_tag=assistant_tag, testing=False, model_name=None)
    if args.load_rep_reader and os.path.exists(rep_reader_meta):
        print(f"[Phase] Loading cached RepReader from {rep_reader_cache}")
        with open(rep_reader_meta) as f:
            meta = json.load(f)
        from control.repe.rep_readers import PCARepReader, ClusterMeanRepReader, RandomRepReader
        cls_map = {'pca': PCARepReader, 'cluster_mean': ClusterMeanRepReader, 'random': RandomRepReader}
        rr_cls = cls_map[meta['direction_method']]
        rep_reader = rr_cls(**meta.get('init_kwargs', {}))
        rep_reader.directions = {int(k): np.load(os.path.join(rep_reader_cache, f'direction_{k}.npy')) for k in meta['directions_layers']}
        rep_reader.direction_signs = {int(k): np.array(v) for k,v in meta['direction_signs'].items()}
        if 'H_train_means_layers' in meta:
            rep_reader.H_train_means = {int(k): np.load(os.path.join(rep_reader_cache, f'H_mean_{k}.npy')) for k in meta['H_train_means_layers']}
        print("[Phase] RepReader loaded.")
    else:
        rep_reader = rep_reading.get_directions(
            dataset['train']['data'],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            train_labels=dataset['train']['labels'],
            direction_method='pca',
            n_difference=1,
            batch_size=2,
            direction_finder_kwargs={'n_components': 4},
        )
        if args.save_rep_reader:
            print(f"[Phase] Saving RepReader to {rep_reader_cache}")
            meta = {
                'direction_method': rep_reader.direction_method,
                'init_kwargs': {'n_components': rep_reader.n_components} if hasattr(rep_reader, 'n_components') else {},
                'directions_layers': list(rep_reader.directions.keys()),
                'direction_signs': {str(k): list(v) for k,v in rep_reader.direction_signs.items()} if rep_reader.direction_signs is not None else {},
            }
            if hasattr(rep_reader, 'H_train_means') and getattr(rep_reader, 'H_train_means', None):
                meta['H_train_means_layers'] = list(rep_reader.H_train_means.keys())
            with open(rep_reader_meta, 'w') as f: json.dump(meta, f, indent=2)
            for layer, arr in rep_reader.directions.items():
                np.save(os.path.join(rep_reader_cache, f'direction_{layer}.npy'), arr)
            if hasattr(rep_reader, 'H_train_means') and getattr(rep_reader, 'H_train_means', None):
                for layer, arr in rep_reader.H_train_means.items():
                    np.save(os.path.join(rep_reader_cache, f'H_mean_{layer}.npy'), arr)
            print("[Phase] RepReader saved.")
    # Select top layers by accuracy evaluation
    from control.experiment_utils import evaluate_repread_accuracy
    plot_dir = ensure_plot_dir('bandwagon_repe_demo_plots_clean')
    results, best_layer = evaluate_repread_accuracy(rep_reading, dataset, rep_token, hidden_layers, rep_reader, plot_dir, debug=args.debug)
    control_layers = sorted(rep_reader.directions.keys(), key=lambda l: results.get(l, 0), reverse=True)[:15]
    rep_control = pipeline("rep-control", model=model, tokenizer=tokenizer, layers=control_layers, control_method="reading_vec")

    # Discover control coefficients dynamically
    from control.repe_experiment import RepEControlExperiment
    print("[Phase] Discovering control coefficients...")
    t0 = time.time()
    coeffs = discover_coeffs(rep_control, tokenizer, rep_reader, control_layers, data_manager, device, debug=args.debug)
    print(f"[Phase] Coefficient discovery completed in {time.time()-t0:.1f}s")
    coeffs = sorted(coeffs)
    full_coeffs = coeffs[:]
    print(f"Discovered coefficients ({len(full_coeffs)}): {np.round(full_coeffs,3)}")
    if args.coeff_limit and args.coeff_limit < len(coeffs):
        # Evenly pick spanning indices including endpoints
        if args.coeff_limit < 2:
            idxs = [0]
        else:
            idxs = np.linspace(0, len(coeffs)-1, num=args.coeff_limit, dtype=int).tolist()
        coeffs = [coeffs[i] for i in sorted(set(idxs))]
        print(f"[Phase] Using coeff subset ({len(coeffs)} of {len(full_coeffs)} spanning range): {np.round(coeffs,3)}")
    else:
        print("[Phase] Using all discovered coefficients (no coeff-limit applied).")

    # Explicit override
    if args.coeffs:
        try:
            override = [float(x) for x in args.coeffs.split(',') if x.strip()]
        except ValueError:
            raise RuntimeError(f"Invalid --coeffs list: {args.coeffs}")
        coeffs = override
        print(f"[Phase] Overriding coefficients with explicit list ({len(coeffs)}): {coeffs}")

    # Build unbiased generation prompts for each feed
    def feed_prompt(feed_posts: List[str]) -> str:
        lines = "\n".join(f"{i+1}) {p}" for i,p in enumerate(feed_posts))
        return (
            f"Pretend you are a Facebook user. You have seen {len(feed_posts)} recent user post(s):\n{lines}\n\n"
            f"Now write your own short post (within 128 tokens).\nPost:"
        )

    repe_experiment = RepEControlExperiment(model, tokenizer, device, RESULTS_DIR, rep_control, rep_reader, control_layers, is_testing_mode=args.debug, num_permutations=1, experiment_suffix="_demo")

    # Sentiment model
    print("[Phase] Loading sentiment model...")
    t_sent0 = time.time()
    sent_tok, sent_cfg, sent_model = sentiment_model(device, fp16=args.fp16_sentiment, compile_model=args.compile_sentiment)
    print(f"[Phase] Sentiment model ready in {time.time()-t_sent0:.1f}s")

    # Already defined earlier

    generations = []
    do_generation = True
    if args.reuse_generation and os.path.exists(raw_gen_path):
        print(f"[Phase] Reusing existing raw generations: {raw_gen_path}")
        with open(raw_gen_path) as f:
            generations = json.load(f)
        do_generation = False
    elif args.resume_generation and os.path.exists(raw_gen_path):
        print(f"[Phase] Resuming generation from existing raw file: {raw_gen_path}")
        with open(raw_gen_path) as f:
            generations = json.load(f)
        do_generation = True  # continue missing
    else:
        print("[Phase] Starting generation (no sentiment scoring yet)...")
        generations = []
        do_generation = True

    if do_generation:
        # Determine pool iteration order early to compute total prompts dynamically
        pool_sequence = [("positive", pos_groups), ("negative", neg_groups)]
        if args.negative_first:
            pool_sequence = [("negative", neg_groups), ("positive", pos_groups)]
        # Allow limiting pools via --pools argument (optional)
        if args.pools:
            requested = [p.strip().lower() for p in args.pools.split(',') if p.strip()]
            pool_sequence = [ps for ps in pool_sequence if ps[0] in requested]
            print(f"[Phase] Using pools: {[p for p,_ in pool_sequence]}")
        num_pools = len(pool_sequence)
        # Count feed prompts including size 0; keep existing pattern using sum(range()) for continuity
        total_prompts = num_pools * sum(range(0, args.max_group_size + 1)) * args.feeds_per_size
        prompt_counter = 0
        t_gen0 = time.time()
        last_flush = time.time()
        smoothing_window = []
        existing_key_set = set()
        if generations:
            for rec in generations:
                existing_key_set.add((rec['pool'], rec['group_size'], rec['group_index'], rec['coeff']))
            if args.resume_generation:
                completed_feeds = 0
                total_feeds = num_pools * sum(range(0, args.max_group_size + 1))
                for pool_name, group_map in pool_sequence:
                    for n in range(0, args.max_group_size + 1):
                        feeds = group_map[n]
                        for feed_idx, _ in enumerate(feeds):
                            if all((pool_name, n, feed_idx, c) in existing_key_set for c in coeffs):
                                completed_feeds += 1
                print(f"[Phase] Resume coverage: {completed_feeds}/{total_feeds} feeds fully done for current coeff subset ({len(coeffs)} coeffs)")
        for pool_name, group_map in pool_sequence:
            for n in range(0, args.max_group_size + 1):
                feeds = group_map[n]
                for feed_idx, feed in enumerate(feeds):  # feeds already limited to feeds_per_size
                    if generations:
                        missing = [c for c in coeffs if (pool_name, n, feed_idx, c) not in existing_key_set]
                        if args.resume_generation and not missing:
                            prompt_counter += 1
                            if args.progress and (prompt_counter == 1 or prompt_counter % args.progress_every == 0):
                                elapsed = time.time() - t_gen0
                                smoothing_window.append(elapsed)
                                if len(smoothing_window) > 5:
                                    smoothing_window.pop(0)
                                rate = prompt_counter / (sum(smoothing_window)/len(smoothing_window))
                                eta = (total_prompts - prompt_counter) / max(rate, 1e-6)
                                print(f"[Progress] Feed {prompt_counter}/{total_prompts} pool={pool_name} n={n} idx={feed_idx} (skip) | {rate:.2f} feeds/s ETA {eta/60:.1f}m")
                            continue
                        control_coeffs_cur = missing if args.resume_generation else coeffs
                    else:
                        control_coeffs_cur = coeffs
                    prompt_counter += 1
                    if args.progress and (prompt_counter == 1 or prompt_counter % args.progress_every == 0):
                        elapsed = time.time() - t_gen0
                        smoothing_window.append(elapsed)
                        if len(smoothing_window) > 5:
                            smoothing_window.pop(0)
                        rate = prompt_counter / (sum(smoothing_window)/len(smoothing_window))
                        eta = (total_prompts - prompt_counter) / max(rate, 1e-6)
                        print(f"[Progress] Feed {prompt_counter}/{total_prompts} pool={pool_name} n={n} idx={feed_idx} | {rate:.2f} feeds/s ETA {eta/60:.1f}m")
                    if not control_coeffs_cur:
                        continue
                    prompt = feed_prompt(feed)
                    gens = repe_experiment.test_time_run(
                        prompts=[prompt],
                        scenario_name=f"bandwagon_{pool_name}_n{n}_g{feed_idx}",
                        operator='linear_comb',
                        control_coeffs=control_coeffs_cur,
                        dynamic_discovery=False,
                        max_new_tokens=args.max_new_tokens,
                        batch_size=1,
                        return_full_text=False,
                        debug=args.debug
                    )
                    for g in gens:
                        record = {
                            'pool': pool_name,
                            'group_size': n,
                            'group_index': feed_idx,
                            'coeff': g['coeff'],
                            'generation': g['generation']
                        }
                        generations.append(record)
                        existing_key_set.add((pool_name, n, feed_idx, g['coeff']))
                    if args.flush_every and (prompt_counter % args.flush_every == 0 or (time.time() - last_flush) > 60):
                        with open(raw_gen_path, 'w') as f:
                            json.dump(generations, f, ensure_ascii=False, indent=2)
                        last_flush = time.time()
                        if args.progress:
                            print(f"[Progress] Flushed {len(generations)} generations to disk.")
        print(f"[Phase] Generation complete in {time.time()-t_gen0:.1f}s. Records: {len(generations)}")
        with open(raw_gen_path, 'w') as f:
            json.dump(generations, f, ensure_ascii=False, indent=2)
        print(f"Saved raw generations (no scores): {raw_gen_path}")

    # Sentiment Scoring Phase (separate)
    if os.path.exists(scored_path) and not args.rescore:
        print(f"[Phase] Scored file exists and --rescore not set: {scored_path}")
        with open(scored_path) as f:
            scored_records = json.load(f)
    else:
        print("[Phase] Scoring sentiments in batches...")
        scored_records = []
        batch_size = args.score_batch_size
        t_score0 = time.time()
        for i in range(0, len(generations), batch_size):
            batch = generations[i:i+batch_size]
            texts = [r['generation'] for r in batch]
            scores = batch_sentiment_scores(texts, sent_tok, sent_cfg, sent_model, device)
            for rec, s in zip(batch, scores):
                new_rec = dict(rec)
                new_rec['sentiment_score'] = s
                scored_records.append(new_rec)
            if args.progress and (i // batch_size == 0 or (i // batch_size + 1) % args.progress_every == 0):
                done = i + len(batch)
                rate = done / max(time.time()-t_score0, 1e-6)
                eta = (len(generations) - done) / max(rate, 1e-6)
                print(f"[Progress][Scoring] {done}/{len(generations)} gens | {rate:.1f} gen/s ETA {eta/60:.1f}m")
        print(f"[Phase] Sentiment scoring complete in {time.time()-t_score0:.1f}s")
        with open(scored_path, 'w') as f:
            json.dump(scored_records, f, ensure_ascii=False, indent=2)
        print(f"Saved scored generations: {scored_path}")

    # Aggregate & plot from scored records
    import pandas as pd
    df = pd.DataFrame(scored_records)
    summary = df.groupby(['pool','group_size','coeff']).sentiment_score.agg(['mean','std','count']).reset_index()
    summary.rename(columns={'mean':'mean_score','std':'std_score','count':'n'}, inplace=True)
    csv_path = os.path.join(RESULTS_DIR, 'sentiment_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")
    plot(summary, coeffs)
    if getattr(args, 'negative_custom_plot', False):
        # Coefficient mapping per spec (subset) if available in summary
        coeff_order = [-0.625, -0.247, 0.0, 0.43359375, 0.736]
        cbi_vals = [2.55, 2.80, 3.00, 3.11, 3.13]
        coeff_map = {c: cbi for c, cbi in zip(coeff_order, cbi_vals)}
        import pandas as pd  # ensure pd alias in scope for function
    plot_negative_custom(summary, coeff_map, coeff_order, RESULTS_DIR, comic_sans_ttf=getattr(args, 'comic_sans_ttf', ''))


def plot(summary_df, coeffs):
    coeffs_sorted = sorted(set(coeffs))
    fig, axes = plt.subplots(1,2, figsize=(14,5), sharey=True)
    for ax, pool in zip(axes, ['positive','negative']):
        pool_df = summary_df[summary_df.pool == pool]
        for c in coeffs_sorted:
            sub = pool_df[pool_df.coeff == c]
            ax.plot(sub.group_size, sub.mean_score, marker='o', label=f'{c:+.2f}')
        ax.set_title(f'{pool.capitalize()} feed')
        ax.set_xlabel('Feed size (n)')
        if pool == 'positive': ax.set_ylabel('Sentiment (pos - neg)')
        ax.grid(alpha=0.3)
    axes[1].legend(title='Coeff', bbox_to_anchor=(1.04,1), loc='upper left')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'sentiment_curves.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def plot_negative_custom(summary_df, coeff_map, coeff_order, out_dir, comic_sans_ttf: str = ""):
    """Custom negative-only plots:
    1) Sentiment vs Number of Negative Posts (group_size) with curves for CBI values.
    2) Sentiment vs CBI with curves for different group_size values.
    Styling per user spec.
    """
    import matplotlib as mpl
    import pandas as pd  # needed for Categorical mapping when called from plot-only path
    # Filtering & mapping
    neg_df = summary_df[summary_df.pool == 'negative'].copy()
    # Keep only specified coefficients
    neg_df = neg_df[neg_df.coeff.isin(coeff_order)]
    if neg_df.empty:
        print("[CustomPlot] No matching negative records for provided coefficients.")
        return
    # Map coefficients to CBI
    neg_df['CBI'] = neg_df['coeff'].map(coeff_map)
    # Enforce ordering categories
    neg_df['CBI'] = pd.Categorical(neg_df['CBI'], [coeff_map[c] for c in coeff_order], ordered=True)

    # Style settings with font fallback
    palette = ["#A2D94D", "#22BDD2", "#859ED7", "#9368AB", "#F8A87B"]
    from matplotlib import font_manager
    def ensure_comic_sans(ttf_path: str):
        if ttf_path and os.path.isfile(ttf_path):
            try:
                font_manager.fontManager.addfont(ttf_path)
                font_manager._rebuild()
            except Exception as e:
                print(f"[CustomPlot] Failed adding provided Comic Sans TTF '{ttf_path}': {e}")
        available = {f.name for f in font_manager.fontManager.ttflist}
        return 'Comic Sans MS' in available

    have_cs = 'Comic Sans MS' in {f.name for f in font_manager.fontManager.ttflist}
    if not have_cs:
        # Try provided path
        if comic_sans_ttf:
            print(f"[CustomPlot] Attempting to load Comic Sans from provided path: {comic_sans_ttf}")
            have_cs = ensure_comic_sans(comic_sans_ttf)
        # Try common system locations if still missing
        if not have_cs:
            common_paths = [
                '/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf',
                '/usr/share/fonts/truetype/msttcorefonts/comic.ttf',
                '/usr/share/fonts/truetype/msttcorefonts/comicbd.ttf',
                os.path.expanduser('~/.local/share/fonts/Comic_Sans_MS.ttf'),
                os.path.expanduser('~/.fonts/Comic_Sans_MS.ttf')
            ]
            for pth in common_paths:
                if os.path.isfile(pth):
                    print(f"[CustomPlot] Attempting to load Comic Sans from {pth}")
                    have_cs = ensure_comic_sans(pth)
                    if have_cs:
                        break
    if have_cs:
        mpl.rcParams['font.family'] = 'Comic Sans MS'
    else:
        if not getattr(plot_negative_custom, '_font_warned', False):
            print("[CustomPlot] Comic Sans MS not available. Install via: sudo apt-get update && sudo apt-get install -y ttf-mscorefonts-installer (enable multiverse). Using DejaVu Sans.")
            plot_negative_custom._font_warned = True
        mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['font.size'] = 17
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    # Figure 1: curves across group_size for each CBI
    fig1, ax1 = plt.subplots(figsize=(4.8,3.6))
    for i, c in enumerate(coeff_order):
        cbi = coeff_map[c]
        sub = neg_df[neg_df.coeff == c].sort_values('group_size')
        ax1.plot(sub.group_size, sub.mean_score, marker='o', label=f'CBI {cbi}', color=palette[i % len(palette)])
    ax1.set_xlabel('Number of Negative Posts', fontsize=17, fontweight='bold')
    ax1.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Remove title per spec
    # Bold all text except tick labels: ticks normal size 12
    for tick in ax1.xaxis.get_major_ticks():
        lbl = getattr(tick, 'label1', None) or getattr(tick, 'label', None)
        if lbl:
            lbl.set_fontsize(12)
            lbl.set_fontweight('normal')
    for tick in ax1.yaxis.get_major_ticks():
        lbl = getattr(tick, 'label1', None) or getattr(tick, 'label', None)
        if lbl:
            lbl.set_fontsize(12)
            lbl.set_fontweight('normal')
    leg1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5,-0.28), ncol=3, frameon=False, fontsize=12)
    for txt in leg1.get_texts():
        txt.set_fontweight('bold')
    fig1.tight_layout(rect=[0,0.05,1,1])
    out1 = os.path.join(out_dir, 'negative_cbi_vs_group_size.png')
    fig1.savefig(out1, dpi=200)
    print(f"Saved custom plot (group size curves): {out1}")

    # Figure 2: curves across CBI for each group size
    fig2, ax2 = plt.subplots(figsize=(4.8,3.6))
    # Determine group sizes present
    gs_list = sorted(neg_df.group_size.unique())
    # Use a colormap for group sizes if more than palette; else reuse palette
    cmap = mpl.cm.get_cmap('viridis')
    for idx, g in enumerate(gs_list):
        gsub = neg_df[neg_df.group_size == g].copy()
        gsub = gsub.set_index('coeff').reindex(coeff_order).reset_index()
        ax2.plot([coeff_map[c] for c in coeff_order], gsub.mean_score, marker='o', label=f'n={g}', color=cmap(idx / max(len(gs_list)-1,1)))
    ax2.set_xlabel('CBI', fontsize=17, fontweight='bold')
    ax2.set_ylabel('Sentiment Score', fontsize=17, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for tick in ax2.xaxis.get_major_ticks():
        lbl = getattr(tick, 'label1', None) or getattr(tick, 'label', None)
        if lbl:
            lbl.set_fontsize(12)
            lbl.set_fontweight('normal')
    for tick in ax2.yaxis.get_major_ticks():
        lbl = getattr(tick, 'label1', None) or getattr(tick, 'label', None)
        if lbl:
            lbl.set_fontsize(12)
            lbl.set_fontweight('normal')
    leg2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5,-0.28), ncol=4, frameon=False, fontsize=12)
    for txt in leg2.get_texts():
        txt.set_fontweight('bold')
    fig2.tight_layout(rect=[0,0.07,1,1])
    out2 = os.path.join(out_dir, 'negative_cbi_vs_coeff.png')
    fig2.savefig(out2, dpi=200)
    print(f"Saved custom plot (CBI curves): {out2}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Model name/path (compatible with unified bias loader)')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--max-group-size', type=int, default=10, help='Maximum group size (n) to generate (1..N)')
    p.add_argument('--feeds-per-size', type=int, default=10, help='Number of random feeds per group size')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--fp16-sentiment', action='store_true', help='Load sentiment model in fp16 (GPU)')
    p.add_argument('--compile-sentiment', action='store_true', help='torch.compile the sentiment model (PyTorch 2+)')
    p.add_argument('--progress', action='store_true', help='Print periodic progress lines during generation')
    p.add_argument('--progress-every', type=int, default=10, help='Feed prompt interval for progress prints')
    p.add_argument('--reuse-generation', action='store_true', help='Skip generation phase and reuse existing raw generations if present')
    p.add_argument('--resume-generation', action='store_true', help='Resume generation: continue missing coefficient combos (ignored if --reuse-generation)')
    p.add_argument('--rescore', action='store_true', help='Force re-scoring even if scored file exists')
    p.add_argument('--score-batch-size', type=int, default=64, help='Batch size for sentiment scoring')
    p.add_argument('--save-rep-reader', action='store_true', help='Persist trained RepReader directions to disk')
    p.add_argument('--load-rep-reader', action='store_true', help='Load cached RepReader if available (skip retraining)')
    p.add_argument('--quiet-hf', action='store_true', help='Suppress Hugging Face transformer warnings')
    p.add_argument('--flush-every', type=int, dest='flush_every', default=0, help='Flush raw generations to disk every N feed prompts (0=only end)')
    p.add_argument('--coeff-limit', type=int, default=0, help='If >0 select this many coefficients spanning full discovered range (include endpoints)')
    p.add_argument('--coeffs', type=str, default='', help='Comma-separated explicit coefficient list (overrides discovery subset & --coeff-limit)')
    p.add_argument('--negative-first', action='store_true', help='Iterate negative feed pool before positive pool')
    p.add_argument('--pools', type=str, default='', help='Comma-separated subset of pools to use (positive,negative). Empty=both')
    p.add_argument('--test-mode', action='store_true', help='Shortcut: limit sizes and feeds (overrides to small values unless already smaller)')
    p.add_argument('--plot-only', action='store_true', help='Only load existing scored or summary data and regenerate plot')
    p.add_argument('--negative-custom-plot', action='store_true', help='Also produce custom negative-only CBI plots')
    p.add_argument('--comic-sans-ttf', type=str, default='', help='Path to Comic Sans MS .ttf file to load if system font unavailable')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
