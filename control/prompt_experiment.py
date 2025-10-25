from control.base import ControlExperiment
from control.env_config import REASONING_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE
import torch
import torch.nn.functional as F # For softmax

def get_prompt_based_choice_probabilities(model, tokenizer, prompt_text, target_choice_tokens_map, device, temperature=1.0, debug=False):
    """
    Gets probabilities for target choices using a standard forward pass on the base model.
    This is used for prompt-based control where the control is in the text itself.
    """
    if debug:
        print(f"[DEBUG] Getting prompt-based probabilities...")
        print(f"[DEBUG]   Prompt ending: '...{prompt_text[-40:]}'")

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :].squeeze()
    # Apply temperature scaling before softmax
    if temperature != 1.0:
        logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)

    choice_probs = {}
    for choice_label, token_ids in target_choice_tokens_map.items():
        if not token_ids:
            choice_probs[choice_label] = 0.0
            continue
        p_tensor = torch.tensor(token_ids, device=probabilities.device, dtype=torch.long)
        choice_probs[choice_label] = probabilities[p_tensor].sum().item()

    if debug:
        top_k_probs, top_k_indices = torch.topk(probabilities, 10)
        print("[DEBUG]   --- Top 10 Most Likely Next Tokens (Prompt-Based) ---")
        for i in range(10):
            token_id = top_k_indices[i].item()
            prob = top_k_probs[i].item()
            decoded_token = repr(tokenizer.decode([token_id]))
            print(f"[DEBUG]     {i+1}. Token: {decoded_token:<15} | Probability: {prob:.4f}")
        print("[DEBUG]   ------------------------------------")
        print(f"[DEBUG]   Target Choice Probs: { {k: f'{v:.3f}' for k,v in choice_probs.items()} }")

    return choice_probs

class PromptControlExperiment(ControlExperiment):
    """Runs prompt-based control experiments with fine-grained Likert scale control."""
    def __init__(self, model, tokenizer, device, plot_dir, bias_type="authority", likert_levels=None, is_testing_mode=False, num_permutations=3, reasoning_temperature=1.0, choice_temperature=1.0, experiment_suffix=""):
        super().__init__(model, tokenizer, device, plot_dir, is_testing_mode, num_permutations, experiment_suffix)
        # Store bias type for prompt generation
        self.bias_type = bias_type.lower()
        # Use percentage scale (0-100) with step size 5 - all levels even in test mode
        self.likert_levels = likert_levels if likert_levels is not None else list(range(0, 101, 5))  # [0, 5, 10, 15, ..., 100]
        self.minimum = min(self.likert_levels) if likert_levels else 0
        self.maximum = max(self.likert_levels) if likert_levels else 100
        self.reasoning_batch_size = REASONING_BATCH_SIZE  # Batch size for generating reasoning text loaded from .env
        # Temperature settings for prompt-based experiments
        self.reasoning_temperature = reasoning_temperature
        self.choice_temperature = choice_temperature

    def _generate_likert_control_prompt(self, level):
        """
        Generate a bias-specific Likert scale control prompt based on the level and bias type.
        Level min: No bias
        Level max: Maximum bias
        """
        bias_descriptions = {
            'authority': {
                'max_description': 'always trust authority figures',
                'min_description': 'never trust authority figures'
            },
            'bandwagon': {
                'max_description': 'always follow what the majority does',
                'min_description': 'never follow what the majority does'
            },
            'framing': {
                'max_description': 'always be strongly influenced by how information is presented',
                'min_description': 'never be influenced by how information is presented'
            },
            'confirmation': {
                'max_description': 'always seek only information that confirms your existing beliefs',
                'min_description': 'never seek only information that confirms your existing beliefs'
            }
        }
        
        if self.bias_type not in bias_descriptions:
            # Fallback to authority bias for unknown types
            bias_type = 'authority'
        else:
            bias_type = self.bias_type
            
        desc = bias_descriptions[bias_type]
        return f"Pretend you have {bias_type} bias at level {level}% out of 100% (where 0%={desc['min_description']}, 100%={desc['max_description']}). Each level increases bias by exactly 5%. Act with this precise level {level}% of {bias_type} bias."

    def _get_prompt_choice_probabilities(self, prompts, batch_mode, mcq_prompt_addon, target_token_map, level=None, mcq_scenario_key='', **kwargs):
        """Wrapper for the prompt-based probability calculation function. Can operate in batch or single mode."""

        # BATCH MODE: Generate reasoning for all prompts that need it.
        if batch_mode:
            prompts_for_gen = []
            if mcq_scenario_key not in ["reasoning_then_choice", "thinking_first"]:
                return

            for p_data in prompts:
                control_prompt = self._generate_likert_control_prompt(p_data['level'])
                full_mcq_prompt = f"{control_prompt}\n\n{p_data['base_prompt']}{p_data['options_block']}\n{mcq_prompt_addon}"
                prompt_for_reasoning = f"{self.user_tag}{full_mcq_prompt}{self.thinking_assistant_tag if mcq_scenario_key == 'thinking_first' else self.assistant_tag}"
                prompts_for_gen.append(prompt_for_reasoning)

            # Process in smaller batches
            for i in range(0, len(prompts), self.reasoning_batch_size):
                batch_prompts_for_gen = prompts_for_gen[i:i+self.reasoning_batch_size]
                outputs = self.model.generate(
                    self.tokenizer(batch_prompts_for_gen, return_tensors="pt", padding=True).to(self.device),
                    max_new_tokens=MAX_NEW_TOKENS,  # Load from .env
                    do_sample=True,  # Enable sampling for temperature
                    temperature=self.reasoning_temperature,
                    pad_token_id=self.tokenizer.eos_token_id, 
                    eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>" if "qwen" in self.model.config.name_or_path.lower() else self.tokenizer.eos_token_id)
                )
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for j, p_data in enumerate(prompts[i:i+self.reasoning_batch_size]):
                    full_generated_text = generated_texts[j]
                    p_data['reasoning_text'] = full_generated_text.replace(batch_prompts_for_gen[j], "").strip()
            return

        # SINGLE MODE: Calculate probability for a single, fully-formed prompt.
        p_data = prompts[0]
        base_prompt = p_data['base_prompt']
        options_block = p_data['options_block']
        level = p_data['level']
        control_prompt = self._generate_likert_control_prompt(level)
        full_mcq_prompt = f"{control_prompt}\n\n{base_prompt}{options_block}\n{mcq_prompt_addon}"
        reasoning_text = p_data.get('reasoning_text', "")

        if mcq_scenario_key in ["choice_first", "choice_then_reasoning"]:
            prompt_for_prob_calc = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}{self.assistant_prompt_for_choice}"
        elif mcq_scenario_key in ["reasoning_then_choice", "thinking_first"]:
            prompt_for_reasoning = f"{self.user_tag}{full_mcq_prompt}{self.thinking_assistant_tag if mcq_scenario_key == 'thinking_first' else self.assistant_tag}"
            prompt_for_prob_calc = f"{prompt_for_reasoning}{reasoning_text}\n\nAnswer: "
        else:
            raise ValueError(f"Unknown mcq_scenario_key: {mcq_scenario_key}")

        choice_probs = get_prompt_based_choice_probabilities(self.model, self.tokenizer, prompt_for_prob_calc, target_token_map, self.device, temperature=self.choice_temperature, debug=self.is_testing_mode)
        return choice_probs, reasoning_text

    def run(self, scenarios, mcq_options, prompt_template, scenario_name):
        print("\n" + "="*50 + f"\n--- Running Fine-Grained Prompt-Based Control on {scenario_name} Scenarios ---\n" + "="*50)

        choice_labels = list(mcq_options.keys())
        option_to_number_map = {label: str(i+1) for i, label in enumerate(choice_labels)}
        target_token_map = self._get_target_token_map(choice_labels, option_to_number_map)

        for mcq_key, mcq_instruction in self.mcq_scenarios.items():
            print(f"\n\n{'='*15} TESTING MCQ SCENARIO: {mcq_key} {'='*15}")
            
            avg_probs, detailed_results = self._run_analysis(
                scenarios, mcq_options, prompt_template, self.likert_levels, target_token_map,
                scenario_name, self._get_prompt_choice_probabilities,
                mcq_addon=mcq_instruction,
                mcq_scenario_key=mcq_key
            )

            # Save detailed results to JSON
            self.save_results_to_json(detailed_results, scenario_name, "prompt_likert", mcq_key)
            self.save_plot_data(avg_probs, self.likert_levels, choice_labels, scenario_name, "prompt_likert", mcq_key)

            plot_title = f"Prob of {scenario_name} Choices vs. Percentage {self.bias_type.title()} Bias\\n(Avg. over {self.num_permutations} Permutations, MCQ: {mcq_key})"
            xlabel = f"{self.bias_type.title()} Bias Level (0%=No Bias, 100%=Maximum Bias)"
            xticks_labels = [str(level) for level in self.likert_levels]
            self._plot_results(avg_probs, self.likert_levels, choice_labels, scenario_name, "prompt_likert", plot_title, xlabel, xticks_labels=xticks_labels, legend_options=mcq_options, mcq_scenario_key=mcq_key)
            self._plot_probability_sum(avg_probs, self.likert_levels, choice_labels, scenario_name, "prompt_likert", mcq_key)
            self._plot_likert_score(avg_probs, self.likert_levels, choice_labels, scenario_name, "prompt_likert", mcq_key)
