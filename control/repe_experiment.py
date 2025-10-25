import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F # For softmax
import random
from control.base import ControlExperiment # Assuming this is part of your project structure
from control.env_config import REASONING_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE, STEP_SIZE

# Corrected and Enhanced Helper Functions

def get_controlled_choice_probabilities(rep_control_pipeline, tokenizer, prompt_text, activations, target_choice_tokens_map, device, debug=False, operator='linear_comb', temperature=1.0):
    """
    Gets probabilities for target choices by using the pipeline's internal wrapped_model
    to manually apply activations during a standard forward pass and get the resulting logits.
    
    Args:
        temperature: Temperature for softmax calculation. Higher values make distribution more uniform,
                    lower values make it more peaked. Default 1.0 means no temperature scaling.
    """
    if debug:
        print(f"[DEBUG] Getting controlled probabilities (operator: {operator}, temperature: {temperature})...")
        print(f"[DEBUG]   Prompt ending: '...{prompt_text[-40:]}'")

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    wrapped_model = rep_control_pipeline.wrapped_model

    # Debug: Check model and activation dtypes
    if debug:
        model_dtype = next(wrapped_model.parameters()).dtype
        print(f"[DEBUG] Model dtype: {model_dtype}")
        print(f"[DEBUG] Device: {device}")
        for layer, activation in activations.items():
            print(f"[DEBUG] Layer {layer} activation dtype: {activation.dtype}, shape: {activation.shape}")

    # Ensure activations match model dtype
    model_dtype = next(wrapped_model.parameters()).dtype
    corrected_activations = {}
    for layer, activation in activations.items():
        if activation.dtype != model_dtype:
            if debug:
                print(f"[DEBUG] Converting activation for layer {layer} from {activation.dtype} to {model_dtype}")
            corrected_activations[layer] = activation.to(model_dtype)
        else:
            corrected_activations[layer] = activation

    with torch.no_grad():
        try:
            if debug:
                print(f"[DEBUG] Calling set_controller with layers: {rep_control_pipeline.layers}")
                print(f"[DEBUG] Block name: {rep_control_pipeline.block_name}")
                print(f"[DEBUG] Operator: {operator}")
            
            wrapped_model.set_controller(rep_control_pipeline.layers, corrected_activations, block_name=rep_control_pipeline.block_name, operator=operator)
            
            if debug:
                print(f"[DEBUG] set_controller successful, running forward pass...")
            
            outputs = wrapped_model(**inputs)
            
            if debug:
                print(f"[DEBUG] Forward pass successful, resetting controller...")
            
            wrapped_model.reset()
            
        except Exception as e:
            if debug:
                print(f"[ERROR] Exception in set_controller or forward pass: {e}")
                print(f"[ERROR] Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
            wrapped_model.reset()  # Ensure we reset even on error
            raise

    logits = outputs.logits[:, -1, :].squeeze()
    
    # Apply temperature scaling to logits before softmax
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
        
        print("[DEBUG]   --- Top 10 Most Likely Next Tokens ---")
        for i in range(10):
            token_id = top_k_indices[i].item()
            prob = top_k_probs[i].item()
            decoded_token = repr(tokenizer.decode([token_id]))
            print(f"[DEBUG]     {i+1}. Token: {decoded_token:<15} | Probability: {prob:.4f}")
        print("[DEBUG]   ------------------------------------")
        print(f"[DEBUG]   Target Choice Probs: { {k: f'{v:.3f}' for k,v in choice_probs.items()} }")
        
    return choice_probs


# In repe_experiment.py
def find_dynamic_control_coeffs(
    rep_control_pipeline, tokenizer, rep_reader, control_layers, 
    target_token_map, device, 
    scenarios, mcq_options, prompt_template,
    mcq_prompt_addon, direction_signs, directions, 
    user_tag, assistant_tag, thinking_assistant_tag, assistant_prompt_for_choice,
    mcq_scenario_key,
    operator,
    choice_temperature=1.0,
    threshold=0.9, max_steps=100, initial_step=1.0, min_step=0.001, debug=False
):
    """
    Dynamically finds control coefficients using a two-phase ADAPTIVE SAMPLING strategy.
    The boundary search uses an adaptive threshold based on the baseline (coeff=0) performance.

    Phase 1: Finds boundaries and runs a coarse grid search.
    Phase 2: Adds more samples in regions of high change (high gradient in Likert score).

    Args:
        ...
        threshold (float): The factor of the baseline (coeff=0) probability sum to use
                                  as the cutoff for boundary searching. Defaults to 0.9 (90%).
        ...
    """
    if debug:
        print("\n[DEBUG] Starting find_dynamic_control_coeffs with ADAPTIVE SAMPLING")
        print(f"[DEBUG]   For Operator: '{operator}', MCQ Scenario: '{mcq_scenario_key}'")

    # --- Setup ---
    n_probe_samples = 5
    probe_data = []
    for _ in range(n_probe_samples):
        s = random.choice(scenarios)
        base_prompt_sample = prompt_template.format(**s)
        option_keys = list(mcq_options.keys())
        random.shuffle(option_keys)
        shuffled_options_dict = {k: mcq_options[k] for k in option_keys}
        options_block_sample = "\nOptions:\n" + "\n".join([f"{k}: {shuffled_options_dict[k]}" for k in option_keys])
        probe_data.append((base_prompt_sample, options_block_sample))

    probe_cache = {}
    choice_labels = list(mcq_options.keys())
    likert_weights = {label: 4 - i for i, label in enumerate(sorted(choice_labels))}

    def probe(coeff):
        """Probes a coefficient and returns both (prob_sum, likert_score)."""
        coeff_key = f"{coeff:.4f}"
        if coeff_key in probe_cache:
            return probe_cache[coeff_key]

        prob_sums, likert_scores = [], []
        for base_prompt_sample, options_block_sample in probe_data:
            # Ensure consistent dtype for activations based on model's dtype
            model_dtype = next(rep_control_pipeline.wrapped_model.parameters()).dtype
            if debug:
                print(f"[DEBUG] probe() coeff={coeff:.4f}, model_dtype={model_dtype}")
            
            activations = {
                layer: torch.tensor(
                    coeff * direction_signs[layer][0] * directions[layer][0]
                ).to(device).to(model_dtype)  # Match model's dtype
                for layer in control_layers
            }
            
            if debug:
                for layer, activation in activations.items():
                    print(f"[DEBUG] probe() layer {layer}: activation dtype={activation.dtype}, shape={activation.shape}")
            full_mcq_prompt = f"{base_prompt_sample}{options_block_sample}\n{mcq_prompt_addon}"
            
            if mcq_scenario_key in ["choice_first", "choice_then_reasoning"]:
                prompt_for_prob_calc = f"{user_tag}{full_mcq_prompt}{assistant_tag}{assistant_prompt_for_choice}"
            else: # Fallback for reasoning-first scenarios
                prompt_for_prob_calc = f"{user_tag}{full_mcq_prompt}{assistant_tag}\n\nAnswer: "

            choice_probs = get_controlled_choice_probabilities(rep_control_pipeline, tokenizer, prompt_for_prob_calc, activations, target_token_map, device, debug=False, operator=operator, temperature=choice_temperature)
            prob_sums.append(sum(choice_probs.values()))
            
            total_prob = sum(choice_probs.values())
            if total_prob > 0:
                score = sum(p / total_prob * likert_weights.get(l, 0) for l, p in choice_probs.items())
                likert_scores.append(score)

        result = (np.mean(prob_sums), np.mean(likert_scores) if likert_scores else 2.0)
        probe_cache[coeff_key] = result
        return result

    # --- Calculate Adaptive Threshold ---
    # if the method is orthogonalize, we only do positive coefficients
    if operator == 'orthognalize':
        min_coeff = 0.001
        baseline_prob_sum, _ = probe(min_coeff)
    else:
        baseline_prob_sum, _ = probe(0.0)
    adaptive_threshold = baseline_prob_sum * threshold

    if debug:
        print(f"\n[DEBUG]   Baseline prob sum (coeff=0): {baseline_prob_sum:.4f}")
        print(f"[DEBUG]   Using adaptive threshold for boundary search: {adaptive_threshold:.4f} ({threshold*100:.0f}% of baseline)")

    def _probe_one_direction(start_coeff, step_direction, threshold):
        """Finds boundary using adaptive step size based on probability sum."""
        coeff, best_coeff, step = start_coeff, start_coeff, initial_step
        for _ in range(max_steps):
            next_probe_coeff = coeff + step_direction * step
            prob_sum, _ = probe(next_probe_coeff)
            print(f"[DEBUG] Probing coeff={next_probe_coeff:.4f} -> Prob Sum: {prob_sum:.4f}")
            if prob_sum >= threshold:
                coeff = next_probe_coeff
                best_coeff = coeff
            else:
                step /= 2.0
                if step < min_step: break
        return best_coeff

    # --- Phase 1: Coarse Grid Exploration ---
    if debug: print("\n[DEBUG] Phase 1: Finding boundaries and running coarse grid...")
    max_coeff = _probe_one_direction(0.0, +1, adaptive_threshold)
    min_coeff = _probe_one_direction(0.0, -1, adaptive_threshold)
    # if operator == 'orthognalize':
    #     min_coeff = 0.001
    
    
    # Create the initial coarse grid, ensuring min, max, and zero are included
    n_initial_points = 15
    initial_coeffs = set([min_coeff, max_coeff, 0.0])
    if operator == 'orthognalize':
        initial_coeffs = set([min_coeff, max_coeff])
    initial_coeffs.update(np.linspace(min_coeff, max_coeff, n_initial_points))
    
    # Store the full (prob_sum, likert_score) tuple
    probed_results = {c: probe(c) for c in sorted(list(initial_coeffs))}
    
    if debug:
        print(f"[DEBUG]   Initial coarse grid results:")
        print(f"[DEBUG]     {'Coeff':>8} | {'Prob Sum':>10} | {'Likert Score':>12}")
        print(f"[DEBUG]     {'-'*8} | {'-'*10} | {'-'*12}")
        for c, (p_sum, l_score) in sorted(probed_results.items()):
            print(f"[DEBUG]     {c:8.3f} | {p_sum:10.4f} | {l_score:12.4f}")

    # --- Phase 2: Adaptive Sampling by Largest Value Gap ---
    n_adaptive_samples = 10 # How many additional points to add
    if debug: print(f"\n[DEBUG] Phase 2: Adding {n_adaptive_samples} adaptive samples to largest Likert score gaps...")

    for i in range(n_adaptive_samples):
        sorted_coeffs = sorted(probed_results.keys())
        if len(sorted_coeffs) < 2: break
        
        # Find the interval with the largest gap in Likert scores
        max_gap, best_midpoint, c_interval = -1, None, None
        for j in range(len(sorted_coeffs) - 1):
            c1, c2 = sorted_coeffs[j], sorted_coeffs[j+1]
            s1 = probed_results[c1][1] # Index [1] to get Likert score
            s2 = probed_results[c2][1] # Index [1] to get Likert score
            gap = abs(s2 - s1)
            if gap > max_gap:
                max_gap = gap
                best_midpoint = (c1 + c2) / 2.0
                c_interval = (c1, c2)
        
        if best_midpoint is None or best_midpoint in probed_results: break

        # Add the new probe result to the dictionary
        new_result = probe(best_midpoint)
        probed_results[best_midpoint] = new_result

        if debug:
            c1, c2 = c_interval
            print(f"[DEBUG]   Sample {i+1:2d}: Gap of {max_gap:.3f} found between {c1:.3f} and {c2:.3f}. "
                  f"Adding point @ {best_midpoint:.3f} (ProbSum={new_result[0]:.3f}, Score={new_result[1]:.3f})")

    # --- Finalize the Coefficient List ---
    final_control_coeffs = np.array(sorted(probed_results.keys()))

    if debug:
        print("\n[DYNAMIC] Determined Final Coefficient Set via Adaptive Sampling:")
        print(f"[DYNAMIC]   min_coeff={min_coeff:.3f}, max_coeff={max_coeff:.3f}")
        print(f"[DYNAMIC]   Total points: {len(final_control_coeffs)}")
        print(f"[DYNAMIC]   Final Coeffs: {np.round(final_control_coeffs, 3)}")
        print("[DEBUG] Finished find_dynamic_control_coeffs\n")

    return final_control_coeffs

class RepEControlExperiment(ControlExperiment):
    """Runs RepE-based control experiments."""
    def __init__(self, model, tokenizer, device, plot_dir, rep_control_pipeline, rep_reader, control_layers, is_testing_mode=False, num_permutations=3, experiment_suffix="", reasoning_temperature=1.0, choice_temperature=1.0):
        super().__init__(model, tokenizer, device, plot_dir, is_testing_mode, num_permutations, experiment_suffix)
        self.rep_control_pipeline = rep_control_pipeline
        self.rep_reader = rep_reader
        self.control_layers = control_layers
        self.reasoning_batch_size = REASONING_BATCH_SIZE  # Load from .env
        # Reasoning generation parameters
        self.reasoning_temperature = reasoning_temperature
        self.reasoning_num_samples = 3
        self.reasoning_max_tokens = MAX_NEW_TOKENS  # Load from .env
        # Choice probability calculation parameters
        self.choice_temperature = choice_temperature
        # self.control_coeffs is no longer a class attribute

    def _get_repe_choice_probabilities(self, prompts, batch_mode, mcq_prompt_addon, target_token_map, level=None, operator='linear_comb', mcq_scenario_key='', **kwargs):
        """Wrapper for the probability calculation function. Can operate in batch or single mode."""
        
        if batch_mode:
            if mcq_scenario_key not in ["reasoning_then_choice", "thinking_first"]:
                return

            prompts_by_level = {}
            for p_data in prompts:
                if self.is_testing_mode:
                    scenario = p_data.get('scenario', {})
                    scenario_id = scenario.get('id', 'unknown')
                    print(f"[DEBUG] Processing prompt for scenario {scenario_id}, level {p_data['level']}")
                    print(f"[DEBUG] Scenario keys: {list(scenario.keys())}")
                    if not scenario:
                        print(f"[ERROR] Empty scenario data for prompt!")
                        
                prompts_by_level.setdefault(p_data['level'], []).append(p_data)

            for coeff, level_prompts in tqdm(prompts_by_level.items(), desc="Batch Generating Reasoning"):
                # Get model dtype for consistent tensor operations
                model_dtype = next(self.rep_control_pipeline.wrapped_model.parameters()).dtype
                if self.is_testing_mode:
                    print(f"[DEBUG] Batch mode: coeff={coeff}, model_dtype={model_dtype}")
                
                activations = {
                    layer: torch.tensor(
                        coeff * self.rep_reader.direction_signs[layer][0] * self.rep_reader.directions[layer][0]
                    ).to(self.device).to(model_dtype) 
                    for layer in self.control_layers
                }
                
                if self.is_testing_mode:
                    for layer, activation in activations.items():
                        print(f"[DEBUG] Batch layer {layer}: activation dtype={activation.dtype}, shape={activation.shape}")
                
                for i in range(0, len(level_prompts), self.reasoning_batch_size):
                    batch_prompts_data = level_prompts[i:i+self.reasoning_batch_size]
                    prompts_for_gen = []
                    for p_data in batch_prompts_data:
                        full_mcq_prompt = f"{p_data['base_prompt']}{p_data['options_block']}\n{mcq_prompt_addon}"
                        prompt_for_reasoning = f"{self.user_tag}{full_mcq_prompt}{self.thinking_assistant_tag if mcq_scenario_key == 'thinking_first' else self.assistant_tag}"
                        prompts_for_gen.append(prompt_for_reasoning)

                    if self.is_testing_mode:
                        print(f"[DEBUG] About to generate reasoning for {len(prompts_for_gen)} prompts")
                        print(f"[DEBUG] Requested {self.reasoning_num_samples} samples per prompt")
                        for idx, prompt in enumerate(prompts_for_gen):
                            print(f"[DEBUG] Prompt {idx}: {prompt[:100]}...")
                    
                    # Generate multiple samples by running the pipeline multiple times
                    # since num_return_sequences doesn't work properly in batch mode
                    all_outputs = []
                    for sample_idx in range(self.reasoning_num_samples):
                        if self.is_testing_mode:
                            print(f"[DEBUG] Generating sample {sample_idx + 1}/{self.reasoning_num_samples}")
                        
                        sample_outputs = self.rep_control_pipeline(
                            prompts_for_gen, 
                            activations=activations, 
                            max_new_tokens=self.reasoning_max_tokens, 
                            do_sample=True, 
                            temperature=self.reasoning_temperature, 
                            num_return_sequences=1,  # Force to 1 since batch mode doesn't handle multiple properly
                            pad_token_id=self.tokenizer.eos_token_id, 
                            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"), 
                            batch_size=len(prompts_for_gen)
                        )
                        all_outputs.extend(sample_outputs)

                    if self.is_testing_mode:
                        print(f"[DEBUG] Generated {len(all_outputs)} total outputs across {self.reasoning_num_samples} samples")
                        print(f"[DEBUG] Expected: {len(prompts_for_gen)} * {self.reasoning_num_samples} = {len(prompts_for_gen) * self.reasoning_num_samples}")

                    # Reorganize outputs: group by prompt instead of by sample
                    outputs = []
                    for prompt_idx in range(len(prompts_for_gen)):
                        for sample_idx in range(self.reasoning_num_samples):
                            output_idx = sample_idx * len(prompts_for_gen) + prompt_idx
                            if output_idx < len(all_outputs):
                                outputs.append(all_outputs[output_idx])

                    if self.is_testing_mode:
                        print(f"[DEBUG] Reorganized to {len(outputs)} outputs")
                        print(f"[DEBUG] Output types: {[type(output) for output in outputs[:3]]}")

                    for j, p_data in enumerate(batch_prompts_data):
                        scenario_id = p_data.get('scenario', {}).get('id', 'unknown')
                        # Extract all generated responses for this prompt using the expected samples
                        start_idx = j * self.reasoning_num_samples
                        end_idx = start_idx + self.reasoning_num_samples
                        
                        if self.is_testing_mode:
                            print(f"[DEBUG] Processing scenario {scenario_id}: indices {start_idx}-{end_idx} from {len(outputs)} outputs")
                        
                        # Ensure we don't go beyond the output length
                        if end_idx > len(outputs):
                            end_idx = len(outputs)
                            if start_idx >= len(outputs):
                                error_msg = f"CRITICAL ERROR: No outputs available for scenario {scenario_id} (prompt {j}). This indicates a serious pipeline issue!"
                                print(f"[ERROR] {error_msg}")
                                raise IndexError(error_msg)
                        
                        if start_idx >= end_idx:
                            error_msg = f"CRITICAL ERROR: Invalid index range for scenario {scenario_id}: start_idx={start_idx}, end_idx={end_idx}"
                            print(f"[ERROR] {error_msg}")
                            raise IndexError(error_msg)
                        
                        prompt_outputs = outputs[start_idx:end_idx]
                        
                        # Store all reasoning variants
                        reasoning_variants = []
                        for output in prompt_outputs:
                            try:
                                # Handle different output formats
                                if isinstance(output, dict) and 'generated_text' in output:
                                    full_generated_text = output['generated_text']
                                elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                                    full_generated_text = output[0]['generated_text']
                                else:
                                    # Fallback for unexpected format
                                    full_generated_text = str(output)
                                
                                reasoning_text = full_generated_text.replace(prompts_for_gen[j], "").strip()
                                reasoning_variants.append(reasoning_text)
                            except (KeyError, IndexError, TypeError) as e:
                                if self.is_testing_mode:
                                    print(f"[DEBUG] Error processing output: {e}, output: {output}")
                                reasoning_variants.append("")  # Add empty string as fallback
                        
                        # Ensure we have at least one reasoning variant
                        if not reasoning_variants:
                            reasoning_variants = [""]
                        
                        # Use the first variant as primary, store all variants for potential use
                        p_data['reasoning_text'] = reasoning_variants[0]
                        p_data['reasoning_variants'] = reasoning_variants
                        
                        if self.is_testing_mode:
                            print(f"[DEBUG] Generated {len(reasoning_variants)} reasoning variants for scenario {p_data.get('scenario', {}).get('id', 'unknown')}")
                            for idx, variant in enumerate(reasoning_variants[:2]):  # Show first 2 variants
                                # Show the first 100 characters to check if they're actually different
                                print(f"[DEBUG]   Variant {idx+1}: {variant[:100]}...")
                                if idx == 1 and len(reasoning_variants) > 1:
                                    # Check if variants are actually different
                                    if reasoning_variants[0].strip() == reasoning_variants[1].strip():
                                        print(f"[DEBUG]   WARNING: Variants appear to be identical!")
                                    else:
                                        print(f"[DEBUG]   Good: Variants are different")
                        
            return

        # SINGLE MODE
        p_data = prompts[0]
        coeff = p_data['level']
        
        # Get model dtype for consistent tensor operations
        model_dtype = next(self.rep_control_pipeline.wrapped_model.parameters()).dtype
        if self.is_testing_mode:
            print(f"[DEBUG] Single mode: coeff={coeff}, model_dtype={model_dtype}")
        
        activations = {
            layer: torch.tensor(
                coeff * self.rep_reader.direction_signs[layer][0] * self.rep_reader.directions[layer][0]
            ).to(self.device).to(model_dtype) 
            for layer in self.control_layers
        }
        
        if self.is_testing_mode:
            for layer, activation in activations.items():
                print(f"[DEBUG] Single layer {layer}: activation dtype={activation.dtype}, shape={activation.shape}")
        
        # Use the primary reasoning text (first variant) or fallback to empty
        reasoning_text = p_data.get('reasoning_text', "")
        
        # If we have multiple reasoning variants, we could potentially sample from them
        # For now, we'll use the primary one, but this could be extended for ensemble methods
        if 'reasoning_variants' in p_data and len(p_data['reasoning_variants']) > 1:
            # Could implement sampling strategy here if needed
            # For now, just use the first variant
            reasoning_text = p_data['reasoning_variants'][0]
        
        full_mcq_prompt = f"{p_data['base_prompt']}{p_data['options_block']}\n{mcq_prompt_addon}"

        if mcq_scenario_key in ["choice_first", "choice_then_reasoning"]:
            prompt_for_prob_calc = f"{self.user_tag}{full_mcq_prompt}{self.assistant_tag}{self.assistant_prompt_for_choice}"
        elif mcq_scenario_key in ["reasoning_then_choice", "thinking_first"]:
            prompt_for_reasoning = f"{self.user_tag}{full_mcq_prompt}{self.thinking_assistant_tag if mcq_scenario_key == 'thinking_first' else self.assistant_tag}"
            prompt_for_prob_calc = f"{prompt_for_reasoning}{reasoning_text}\n\nAnswer: "
        else:
            raise ValueError(f"Unknown mcq_scenario_key: {mcq_scenario_key}")

        choice_probs = get_controlled_choice_probabilities(self.rep_control_pipeline, self.tokenizer, prompt_for_prob_calc, activations, target_token_map, self.device, debug=self.is_testing_mode, operator=operator, temperature=self.choice_temperature)
        return choice_probs, reasoning_text

    def run(self, scenarios, mcq_options, prompt_template, scenario_name, operators=['linear_comb','projection']): # ,
        print("\n" + "="*50 + f"\n--- Running RepE Control on {scenario_name} Scenarios ---\n" + "="*50)

        # Debug: Check scenario structure
        if self.is_testing_mode:
            print(f"[DEBUG] Loaded {len(scenarios)} scenarios for {scenario_name}")
            for i, scenario in enumerate(scenarios[:3]):  # Show first 3 scenarios
                print(f"[DEBUG] Scenario {i}: {scenario}")
                print(f"[DEBUG] Scenario keys: {list(scenario.keys())}")
                # Test if the scenario can be formatted with the template
                try:
                    test_prompt = prompt_template.format(**scenario)
                    print(f"[DEBUG] Test prompt: {test_prompt[:100]}...")
                except Exception as e:
                    print(f"[ERROR] Failed to format scenario {i} with template: {e}")

        choice_labels = list(mcq_options.keys())
        option_to_number_map = {label: str(i+1) for i, label in enumerate(choice_labels)}
        target_token_map = self._get_target_token_map(choice_labels, option_to_number_map)

        for mcq_key, mcq_instruction in self.mcq_scenarios.items():
            print(f"\n\n{'='*15} TESTING MCQ SCENARIO: {mcq_key} {'='*15}")
            for operator in operators:
                print(f"\n--- Testing Operator: {operator} ---")

                # --- MODIFIED: Dynamically find coeffs for EACH combination ---
                if not scenarios:
                    print("No scenarios provided. Skipping analysis.")
                    continue
                
                print("Dynamically determining control coefficient range for this operator and MCQ scenario...")
                #  if model name contains gpt we set step size to 5
                step_size = STEP_SIZE
                control_coeffs = find_dynamic_control_coeffs(
                    self.rep_control_pipeline,
                    self.tokenizer,
                    self.rep_reader,
                    self.control_layers,
                    target_token_map,
                    self.device,
                    scenarios=scenarios,
                    mcq_options=mcq_options,
                    prompt_template=prompt_template,
                    mcq_prompt_addon=mcq_instruction, # Use the current MCQ instruction
                    direction_signs=self.rep_reader.direction_signs,
                    directions=self.rep_reader.directions,
                    user_tag=self.user_tag,
                    assistant_tag=self.assistant_tag,
                    thinking_assistant_tag=self.thinking_assistant_tag,
                    assistant_prompt_for_choice=self.assistant_prompt_for_choice,
                    mcq_scenario_key=mcq_key, # Pass the current key
                    operator=operator, # Pass the current operator
                    choice_temperature=self.choice_temperature, # Pass choice temperature
                    threshold=0.9,
                    max_steps=100,
                    initial_step=step_size,
                    min_step=0.001,
                    debug=self.is_testing_mode
                )

                avg_probs, detailed_results = self._run_analysis(
                    scenarios, mcq_options, prompt_template, control_coeffs, target_token_map,
                    scenario_name, self._get_repe_choice_probabilities,
                    mcq_addon=mcq_instruction,
                    operator=operator,
                    mcq_scenario_key=mcq_key
                )

                self.save_results_to_json(detailed_results, scenario_name, f"repe_{operator}", mcq_key)
                self.save_plot_data(avg_probs, control_coeffs, choice_labels, scenario_name, f"repe_{operator}", mcq_key)

                plot_title = f"Prob of {scenario_name} Choices vs. RepE Control ({operator})\\n(Avg. over {self.num_permutations} Permutations, MCQ: {mcq_key})"
                xlabel = f"Control Coefficient (-> More Authoritarian, Operator: {operator})"
                self._plot_results(avg_probs, control_coeffs, choice_labels, scenario_name, f"repe_{operator}", plot_title, xlabel, legend_options=mcq_options, mcq_scenario_key=mcq_key)
                self._plot_probability_sum(avg_probs, control_coeffs, choice_labels, scenario_name, f"repe_{operator}", mcq_key)
                self._plot_likert_score(avg_probs, control_coeffs, choice_labels, scenario_name, f"repe_{operator}", mcq_key)
                self._plot_results_scatter(
                    detailed_results, 
                    avg_probs,
                    control_coeffs,
                    choice_labels,
                    scenario_name,
                    f"repe_{operator}",
                    plot_title,
                    xlabel,
                    legend_options=mcq_options,
                    mcq_scenario_key=mcq_key
                )

                # 2. Call the new function for Likert score scatter
                self._plot_likert_score_scatter(
                    detailed_results, # Pass the detailed results here
                    control_coeffs,
                    choice_labels,
                    scenario_name,
                    f"repe_{operator}",
                    mcq_scenario_key=mcq_key
                )