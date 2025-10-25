import os
import torch
from control.base import ControlExperiment
from control.task_vector_experiment import load_task_vector


class LoRAControlExperiment(ControlExperiment):
    """LoRA task-vector-based control experiment with coefficient range [-2,0]."""

    def __init__(self, model, tokenizer, device, plot_dir, task_vector_path, is_testing_mode=False, num_permutations=5, experiment_suffix=""):
        super().__init__(model, tokenizer, device, plot_dir, is_testing_mode, num_permutations, experiment_suffix)
        self.task_vector_path = task_vector_path
        self.deltas, self.meta_cfg = load_task_vector(task_vector_path)
        self._param_refs = {}
        self._baseline_params = {}
        for name, param in self.model.named_parameters():
            if name in self.deltas:
                self._param_refs[name] = param
                self._baseline_params[name] = param.detach().clone()
        if self.is_testing_mode:
            print(f"[LoRA_CTRL] Tracking {len(self._baseline_params)} params.")

    def _set_scale(self, user_coef: float):
        internal_scale = user_coef
        with torch.no_grad():
            for name, base_w in self._baseline_params.items():
                delta = self.deltas[name].to(base_w.device, base_w.dtype)
                self._param_refs[name].copy_(base_w - internal_scale * delta)

    def _revert(self):
        with torch.no_grad():
            for name, base_w in self._baseline_params.items():
                self._param_refs[name].copy_(base_w)

    def _get_lora_choice_probabilities(self, prompts, batch_mode, mcq_prompt_addon, target_token_map, mcq_scenario_key='', temperature=1.0, **kwargs):
        if batch_mode:
            grouped = {}
            for p in prompts:
                grouped.setdefault(p['level'], []).append(p)
            for coef, plist in grouped.items():
                self._set_scale(coef)
                if self.is_testing_mode:
                    print(f"[LoRA_CTRL] coef {coef:.3f} (int {coef+1:.3f}) prompts {len(plist)}")
                for p in plist:
                    full_mcq_prompt = f"{p['base_prompt']}{p['options_block']}\n{mcq_prompt_addon}"
                    if mcq_scenario_key in ["reasoning_then_choice", "thinking_first"]:
                        prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}\n\nAnswer: "
                    else:
                        prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}{self.assistant_prompt_for_choice}"
                    inputs = self.tokenizer(prompt_for_prob, return_tensors='pt').to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :].squeeze()
                    if temperature != 1.0:
                        logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    choice_probs = {}
                    for label, token_ids in target_token_map.items():
                        if not token_ids:
                            choice_probs[label] = 0.0
                            continue
                        idx = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
                        choice_probs[label] = probs[idx].sum().item()
                    p['_batch_choice_probs'] = choice_probs
                    p['reasoning_text'] = ''
                self._revert()
            return
        p = prompts[0]
        if '_batch_choice_probs' not in p:  # fallback
            coef = p['level']
            self._set_scale(coef)
            full_mcq_prompt = f"{p['base_prompt']}{p['options_block']}\n{mcq_prompt_addon}"
            if mcq_scenario_key in ["reasoning_then_choice", "thinking_first"]:
                prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}\n\nAnswer: "
            else:
                prompt_for_prob = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}{self.assistant_prompt_for_choice}"
            inputs = self.tokenizer(prompt_for_prob, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze()
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            choice_probs = {}
            for label, token_ids in target_token_map.items():
                if not token_ids:
                    choice_probs[label] = 0.0
                    continue
                idx = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
                choice_probs[label] = probs[idx].sum().item()
            self._revert()
            return choice_probs, ''
        return p['_batch_choice_probs'], p.get('reasoning_text', '')

    def run(self, scenarios, mcq_options, prompt_template, scenario_name, control_coeffs, temperature=1.0):
        print("\n" + "="*50 + f"\n--- LoRA Control on {scenario_name} ---\n" + "="*50)
        choice_labels = list(mcq_options.keys())
        option_to_number_map = {label: str(i+1) for i, label in enumerate(choice_labels)}
        target_token_map = self._get_target_token_map(choice_labels, option_to_number_map)
        for mcq_key, mcq_instruction in self.mcq_scenarios.items():
            print(f"\n--- MCQ Scenario: {mcq_key} ---")
            avg_probs, detailed_results = self._run_analysis(
                scenarios, mcq_options, prompt_template, control_coeffs, target_token_map,
                scenario_name, self._get_lora_choice_probabilities,
                mcq_addon=mcq_instruction, mcq_scenario_key=mcq_key, temperature=temperature
            )
            self.save_results_to_json(detailed_results, scenario_name, "lora", mcq_key)
            self.save_plot_data(avg_probs, control_coeffs, choice_labels, scenario_name, "lora", mcq_key)
            title = f"Prob of {scenario_name} Choices vs. LoRA Coef (Avg over {self.num_permutations} perms, MCQ: {mcq_key})"
            xlabel = "LoRA User Coefficient (internal scale = coef + 1)"
            self._plot_results(avg_probs, control_coeffs, choice_labels, scenario_name, "lora", title, xlabel, legend_options=mcq_options, mcq_scenario_key=mcq_key)
            self._plot_probability_sum(avg_probs, control_coeffs, choice_labels, scenario_name, "lora", mcq_key)
            self._plot_likert_score(avg_probs, control_coeffs, choice_labels, scenario_name, "lora", mcq_key)

__all__ = ['LoRAControlExperiment']
