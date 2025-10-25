import os
import torch
import numpy as np
from typing import Dict, Any, List
from control.base import ControlExperiment
from torch.nn import Module
import torch.nn.functional as F


def load_task_vector(task_vector_path: str):
    data = torch.load(task_vector_path, map_location='cpu')
    return data['deltas'], data.get('config', {})


class TaskVectorApplier:
    """Apply and revert a scaled task (concept) vector to a base model in-place."""
    def __init__(self, model: Module, deltas: Dict[str, torch.Tensor]):
        self.model = model
        self.deltas = deltas
        self.original = None

    def apply(self, scale: float):
        if self.original is None:
            self.original = {}
            for name, param in self.model.named_parameters():
                if name in self.deltas:
                    self.original[name] = param.detach().clone()
        with torch.no_grad():
            for name, delta in self.deltas.items():
                if name in dict(self.model.named_parameters()):
                    param = dict(self.model.named_parameters())[name]
                    param.add_(delta.to(param.device, param.dtype) * scale)

    def revert(self):
        if self.original is None:
            return
        with torch.no_grad():
            for name, orig in self.original.items():
                if name in dict(self.model.named_parameters()):
                    param = dict(self.model.named_parameters())[name]
                    param.copy_(orig.to(param.device, param.dtype))


def get_choice_probs(model, tokenizer, prompt_text, target_choice_tokens_map, device, temperature=1.0):
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :].squeeze()
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    choice_probs = {}
    for label, token_ids in target_choice_tokens_map.items():
        if not token_ids:
            choice_probs[label] = 0.0
            continue
        t = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
        choice_probs[label] = probs[t].sum().item()
    return choice_probs


class TaskVectorControlExperiment(ControlExperiment):
    """Evaluate scaling a task vector like a control coefficient sweep."""
    def __init__(self, model, tokenizer, device, plot_dir, task_vector_path, is_testing_mode=False, num_permutations=5, experiment_suffix=""):
        super().__init__(model, tokenizer, device, plot_dir, is_testing_mode, num_permutations, experiment_suffix)
        self.task_vector_path = task_vector_path
        self.deltas, self.meta_cfg = load_task_vector(task_vector_path)
        self.applier = TaskVectorApplier(model, self.deltas)

    def _get_taskvec_choice_probabilities(self, prompts, batch_mode, mcq_prompt_addon, target_token_map, mcq_scenario_key='', temperature=1.0, **kwargs):
        # Only reasoning_then_choice / thinking_first need reasoning generation; mimic prompt experiment logic.
        if batch_mode:
            if mcq_scenario_key not in ["reasoning_then_choice", "thinking_first"]:
                return
            # Group prompts by level (scale) so we apply once per scale.
            by_level = {}
            for p in prompts:
                by_level.setdefault(p['level'], []).append(p)
            for scale, level_prompts in by_level.items():
                if self.is_testing_mode:
                    print(f"[TASKVEC] Applying scale {scale:.3f} to {len(level_prompts)} prompts")
                self.applier.apply(scale)
                try:
                    for p in level_prompts:
                        full_mcq_prompt = f"{p['base_prompt']}{p['options_block']}\n{mcq_prompt_addon}"
                        prompt_for_reasoning = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}"
                        # Single short reasoning generation (optional); keep simple: no multi-sampling now.
                        # Could be expanded mirroring other experiments.
                        p['reasoning_text'] = ''  # placeholder (skipped due to efficiency)
                finally:
                    self.applier.revert()
            return
        # Single mode probability computation
        p = prompts[0]
        scale = p['level']
        self.applier.apply(scale)
        try:
            full_mcq_prompt = f"{p['base_prompt']}{p['options_block']}\n{mcq_prompt_addon}"
            if mcq_scenario_key in ["reasoning_then_choice", "thinking_first"]:
                prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}\n\nAnswer: "
            else:
                prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}{self.assistant_prompt_for_choice}"
            choice_probs = get_choice_probs(self.model, self.tokenizer, prompt_for_prob, target_token_map, self.device, temperature=temperature)
            return choice_probs, ''
        finally:
            self.applier.revert()

    def find_dynamic_scales(
        self,
        scenarios,
        mcq_options,
        prompt_template,
        mcq_prompt_addon,
        target_token_map,
        mcq_scenario_key,
        temperature=1.0,
        threshold=0.9,
        max_steps=100,
        initial_step=1.0,
        min_step=0.01,
        n_initial_points=15,
        n_adaptive_samples=10,
        debug=False,
    ):
        """Dynamically determine a good scale range for the task vector using adaptive probing.

        Strategy mirrors RepE: find min/max scales where probability mass remains reasonable
        (relative to baseline), then refine by adaptive sampling focused on Likert score gaps.
        """
        import random

        if debug:
            print("\n[TASKVEC] Starting dynamic scale probing")

        # Prepare probe data: pick a few random scenarios and shuffle options
        n_probe_samples = 5 if not self.is_testing_mode else min(5, len(scenarios))
        probe_data = []
        for _ in range(n_probe_samples):
            s = random.choice(scenarios)
            base_prompt = prompt_template.format(**s)
            option_keys = list(mcq_options.keys())
            random.shuffle(option_keys)
            shuffled = {k: mcq_options[k] for k in option_keys}
            options_block = "\nOptions:\n" + "\n".join([f"{k}: {shuffled[k]}" for k in option_keys])
            probe_data.append((base_prompt, options_block))

        choice_labels = list(mcq_options.keys())
        likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}

        def probe(scale: float):
            """Return (prob_sum, likert_score) averaged over probe_data at the given scale."""
            prob_sums, likert_scores = [], []
            self.applier.apply(scale)
            try:
                for base_prompt, options_block in probe_data:
                    full_mcq_prompt = f"{base_prompt}{options_block}\n{mcq_prompt_addon}"
                    # Use choice-first style to measure next-token distribution directly
                    if mcq_scenario_key in ["reasoning_then_choice", "thinking_first"]:
                        prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}\n\nAnswer: "
                    else:
                        prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}{self.assistant_prompt_for_choice}"
                    cp = get_choice_probs(self.model, self.tokenizer, prompt_for_prob, target_token_map, self.device, temperature=temperature)
                    psum = sum(cp.values())
                    prob_sums.append(psum)
                    if psum > 0:
                        score = sum((cp.get(l, 0.0) / psum) * likert_weights.get(l, 0) for l in choice_labels)
                        likert_scores.append(score)
            finally:
                self.applier.revert()
            return (float(np.mean(prob_sums)) if prob_sums else 0.0,
                    float(np.mean(likert_scores)) if likert_scores else 2.0)

        # Baseline at scale=0
        baseline_prob_sum, _ = probe(0.0)
        cutoff = baseline_prob_sum * threshold
        if debug:
            print(f"[TASKVEC] Baseline prob sum @0.0 = {baseline_prob_sum:.4f}; threshold={cutoff:.4f}")

        def find_boundary(start_scale: float, direction: int) -> float:
            scale = start_scale
            best = start_scale
            step = initial_step
            for _ in range(max_steps):
                next_scale = scale + direction * step
                psum, _ = probe(next_scale)
                if psum >= cutoff:
                    scale = next_scale
                    best = scale
                else:
                    step /= 2.0
                    if step < min_step:
                        break
            return best

        max_scale = find_boundary(0.0, +1)
        min_scale = find_boundary(0.0, -1)

        # Coarse grid
        initial = set([min_scale, max_scale, 0.0])
        initial.update(np.linspace(min_scale, max_scale, n_initial_points))
        probed = {float(c): probe(float(c)) for c in sorted(initial)}

        # Adaptive refinement by largest Likert gap
        for _ in range(n_adaptive_samples):
            keys = sorted(probed.keys())
            if len(keys) < 2:
                break
            max_gap = -1.0
            mid = None
            for i in range(len(keys) - 1):
                c1, c2 = keys[i], keys[i+1]
                s1, s2 = probed[c1][1], probed[c2][1]
                gap = abs(s2 - s1)
                if gap > max_gap:
                    max_gap = gap
                    mid = (c1 + c2) / 2.0
            if mid is None or mid in probed:
                break
            probed[mid] = probe(mid)

        final_scales = np.array(sorted(probed.keys()))
        if debug:
            print(f"[TASKVEC] Final dynamic scales: {np.round(final_scales, 3)}")
        return final_scales

    def run(self, scenarios, mcq_options, prompt_template, scenario_name, control_coeffs=None, temperature=1.0):
        print("\n" + "="*50 + f"\n--- Task Vector Control on {scenario_name} ---\n" + "="*50)
        choice_labels = list(mcq_options.keys())
        option_to_number_map = {label: str(i+1) for i, label in enumerate(choice_labels)}
        target_token_map = self._get_target_token_map(choice_labels, option_to_number_map)
        # Iterate MCQ scenarios (will include reasoning_then_choice etc.) but user currently uses reasoning_then_choice.
        for mcq_key, mcq_instruction in self.mcq_scenarios.items():
            print(f"\n--- MCQ Scenario: {mcq_key} ---")
            # If no control coeffs provided, compute dynamically per scenario key
            coeffs = control_coeffs
            if coeffs is None or len(coeffs) == 0:
                coeffs = self.find_dynamic_scales(
                    scenarios, mcq_options, prompt_template,
                    mcq_instruction, target_token_map, mcq_key,
                    temperature=temperature,
                    threshold=0.9,
                    debug=self.is_testing_mode,
                )
                if self.is_testing_mode:
                    print(f"[TASKVEC] Using dynamic scales: {np.round(coeffs, 3)}")
            avg_probs, detailed_results = self._run_analysis(
                scenarios, mcq_options, prompt_template, coeffs, target_token_map,
                scenario_name, self._get_taskvec_choice_probabilities,
                mcq_addon=mcq_instruction, mcq_scenario_key=mcq_key
            )
            self.save_results_to_json(detailed_results, scenario_name, "taskvec", mcq_key)
            self.save_plot_data(avg_probs, coeffs, choice_labels, scenario_name, "taskvec", mcq_key)
            title = f"Prob of {scenario_name} Choices vs. Task Vector Scale\\n(Avg. over {self.num_permutations} Permutations, MCQ: {mcq_key})"
            xlabel = "Task Vector Scale Coefficient"
            self._plot_results(avg_probs, coeffs, choice_labels, scenario_name, "taskvec", title, xlabel, legend_options=mcq_options, mcq_scenario_key=mcq_key)
            self._plot_probability_sum(avg_probs, coeffs, choice_labels, scenario_name, "taskvec", mcq_key)
            self._plot_likert_score(avg_probs, coeffs, choice_labels, scenario_name, "taskvec", mcq_key)

__all__ = [
    'TaskVectorControlExperiment', 'load_task_vector', 'TaskVectorApplier'
]
