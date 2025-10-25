# Control Module (Concise Overview)

This folder implements three complementary ways to control cognitive bias in LLM outputs and to evaluate control quality end-to-end.

## What’s here

- Core base and plotting
  - `base.py` — shared experiment orchestration, target token mapping, result aggregation, and plotting (line charts + scatter with dual axis).
- Experiment types
  - `prompt_experiment.py` — prompt-only control via Likert-scale conditioning in the text.
  - `repe_experiment.py` — RepE control via activation editing (coefficients applied to hidden activations; dynamic coefficient search).
  - `api_prompt_experiment.py` — prompt-only control via closed-source models using OpenRouter; frequency-based scoring with multithreading.
- Utilities and config
  - `env_config.py` — loads environment variables (batch sizes, temperatures, API keys, defaults).
  - `experiment_utils.py` — model/tokenizer loading, helper plots and evaluation routines.
  - `evaluate_metrics.py` — computes monotonicity and related metrics and renders annotated plots.

## Control at a glance

- Bias types supported in prompt control: authority, bandwagon, framing, confirmation.
- Likert-scale prompting (prompt and API experiments): level ∈ [0, 100] in 5% steps; text describes min/max bias behaviors.
- MCQ scenarios in `base.py` (mcq_scenarios): variations for ordering “choice” and “reasoning”, e.g.,
  - choice_first, choice_then_reasoning, reasoning_then_choice, thinking_first.
- Target mapping: next-token probabilities are summed over token variants for A–E choices to form choice probabilities.

## RepE (activation) control specifics

- `repe_experiment.py`
  - Forward pass with injected activations to steer outputs; supports operators like `linear_comb` and `projection`.
  - `find_dynamic_control_coeffs(...)`: adaptive two-phase search to identify an effective coefficient range based on baseline probability sum and Likert gradients.
  - Temperature controls for reasoning vs. choice steps; batch reasoning supported via `REASONING_BATCH_SIZE`.

## API prompt control specifics (OpenRouter)

- `api_prompt_experiment.py`
  - Uses OpenRouter Chat Completions; models include GPT-4.x, Claude, Gemini, Mistral, etc. (mapped names inside the file).
  - Samples n completions per prompt and scores by frequency of A–E extraction (robust across response formats).
  - Multithreaded requests via `ThreadPoolExecutor` with `max_workers`.
  - Requires `OPENROUTER_API_KEY` in `.env` (see config below).

## Configuration (env)

Configured via `.env` read by `control/env_config.py`:
- BATCH_SIZE, REASONING_BATCH_SIZE, REP_READING_BATCH_SIZE, BANDWAGON_BATCH_SIZE
- DEFAULT_MODEL (for local HF models), HF_TOKEN
- OPENROUTER_API_KEY (for API experiments)
- NUM_PERMUTATIONS (target token permutations), MAX_NEW_TOKENS, TEMPERATURE, STEP_SIZE

Run `python -m control.env_config` to print the loaded config.

## Metrics and plots

Implemented in `evaluate_metrics.py` and used by `base.py` plotting helpers:
- Monotonicity: NDCG-based global monotonicity score plus Spearman ρ on Likert vs. control coefficient.
- Smoothness: first/second-difference smoothness (for averaged series).
- Efficacy/Intensity: range-based measures of effect and coefficient span.
- Both averaged line plots and raw scatter plots show embedded metrics text boxes; axes rescale control coefficients to [0,1] while annotating actual min/max.

## Outputs

- Plots: saved under `plot_dir/<model_or_api>/...` with multiple views (per-choice probabilities, probability sum, Likert score, scatter variants).
- Data: JSON and CSV-like plot data saved via `save_results_to_json`/`save_plot_data` for post-analysis.

## Minimal usage sketches

- Prompt control (local model):
  1) Load model/tokenizer (see `experiment_utils.load_model_and_tags`).
  2) Create `PromptControlExperiment(model, tokenizer, device, plot_dir, bias_type=...)`.
  3) Call `.run(scenarios, mcq_options, prompt_template, scenario_name)`.

- RepE control (activation editing):
  1) Prepare `rep_control_pipeline`, `rep_reader`, `control_layers`, and tags.
  2) Create `RepEControlExperiment(...)` and run as above. The experiment will search dynamic coefficients and plot results.

- API prompt control (OpenRouter):
  1) Ensure `OPENROUTER_API_KEY` is set in `.env`.
  2) Create `APIPromptControlExperiment(api_key, model_name=..., plot_dir=..., bias_type=...)`.
  3) Call `.run(scenarios, mcq_options, prompt_template, scenario_name)`; set `num_samples_per_prompt` and `max_workers` as needed.

## Notes

- MCQ parsing is robust to formats like “Answer: C”, leading/trailing letters, and standalone A–E tokens.
- Temperatures are separated for reasoning vs. the final choice step to isolate control at decision time.
- When using HF models, ensure tokenizer `pad_token_id` is set (handled in `experiment_utils`).
