#!/usr/bin/env python3
"""
API-based Prompt Control Experiment using OpenRouter for closed-source models.
Uses frequency-based scoring instead of probability-based scoring.
Uses multithreading to parallelize API requests for performance.
"""

import requests
import json
import time
import re
from typing import Dict, List, Optional, Any
from collections import Counter
from control.base import ControlExperiment
from control.env_config import REASONING_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE
import os
from dotenv import load_dotenv

# --- New imports for concurrency and analysis ---
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # Available models for closed-source testing
        self.models = {
            "gpt4": "openai/gpt-4",
            "gpt4-turbo": "openai/gpt-4-turbo",
            "gpt4.1": "openai/gpt-4.1",
            "claude": "anthropic/claude-3.5-sonnet", # Updated model name
            "claude-sonnet-4": "anthropic/claude-sonnet-4",
            "gemini": "google/gemini-pro-1.5",
            "gemini-2.5": "google/gemini-2.5-flash",
            "mistral-large": "mistralai/mistral-nemo",
            "gpt-oss": "openai/gpt-oss-20b",
            "gpt4o": "openai/gpt-4o-mini",
        }
    
    def generate_completion(self, messages: List[Dict], model_name: str = "gpt4", 
                          temperature: float = 0.7, max_tokens: int = 150, **kwargs) -> Optional[List[str]]:
        """
        Generate completion(s) using specified model.
        Can generate multiple completions if 'n' is in kwargs.
        """
        try:
            payload = {
                "model": self.models.get(model_name, self.models["gpt4"]),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            # Add any other kwargs to the payload (e.g., n for multiple samples)
            payload.update(kwargs)
            
            # Increased timeout for potentially longer requests (e.g., with n > 1)
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            # Return a list of all message contents from the 'choices' list
            return [choice["message"]["content"].strip() for choice in result.get("choices", [])]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error generating completion: {e}")
            return None

def extract_choice_from_response(response_text: str, choice_labels: List[str]) -> Optional[str]:
    """
    Extract the choice (A, B, C, D, E) from the model response.
    Supports various response formats.
    """
    if not response_text:
        return None
    
    # Clean the response text
    response_text = response_text.strip().upper()
    
    # Pattern 1: Look for "Answer: X" or "Answer is X"
    answer_pattern = r'ANSWER\s*:?\s*([A-E])'
    match = re.search(answer_pattern, response_text)
    if match:
        choice = match.group(1)
        if choice in choice_labels:
            return choice
    
    # Pattern 2: Look for choice at the end of the response
    if response_text and response_text[-1] in choice_labels:
        return response_text[-1]
    
    # Pattern 3: Look for standalone letter choices (A, B, C, D, E)
    # Search for the choice as a standalone word/character to avoid matching inside words
    for choice in choice_labels:
        if re.search(rf'\b{choice}\b', response_text):
            return choice
            
    # Pattern 4: Look for choice at the beginning of response
    if response_text and response_text[0] in choice_labels:
        return response_text[0]
    
    return None

class APIPromptControlExperiment(ControlExperiment):
    """Runs API-based prompt control experiments with frequency-based scoring for closed-source models."""
    
    def __init__(self, api_key: str, model_name: str = "gpt4", plot_dir: str = "./plots", 
                 bias_type: str = "authority", likert_levels: List[int] = None, 
                 is_testing_mode: bool = False, num_permutations: int = 3, 
                 reasoning_temperature: float = 0.7, choice_temperature: float = 0.1,
                 num_samples_per_prompt: int = 10, max_workers: int = 8): # Increased default workers
        """
        Initialize API-based prompt control experiment.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model to use (e.g., "gpt4", "claude", "gemini")
            num_samples_per_prompt: Number of samples to generate per prompt for frequency calculation
            max_workers: Maximum number of concurrent API requests
        """
        # Initialize base class without model/tokenizer/device (not needed for API)
        self.device = "api"  # Placeholder
        self.plot_dir = plot_dir
        self.is_testing_mode = is_testing_mode
        self.num_permutations = num_permutations
        self.max_workers = max_workers
        
        # API-specific setup
        self.client = OpenRouterClient(api_key)
        self.model_name = model_name
        self.bias_type = bias_type.lower()
        
        # Use percentage scale (0-100) with step size 5
        self.likert_levels = likert_levels if likert_levels is not None else list(range(0, 101, 5))
        self.minimum = min(self.likert_levels) if likert_levels else 0
        self.maximum = max(self.likert_levels) if likert_levels else 100
        
        # Temperature settings
        self.reasoning_temperature = reasoning_temperature
        self.choice_temperature = choice_temperature
        
        # Frequency-based sampling
        self.num_samples_per_prompt = num_samples_per_prompt
        
        # MCQ scenarios - simplified for API
        self.mcq_scenarios = {
            "choice_first": "Please choose the best answer (A, B, C, D, or E):",
        }
        
        # Chat format tags
        self.user_tag = ""  # Not needed for API
        self.assistant_tag = ""  # Not needed for API
        self.assistant_prompt_for_choice = "Answer: "
    
    def _construct_prefilled_messages(self, control_prompt: str, user_content: str, use_prefill_format: bool) -> List[Dict]:
        """Constructs the message list for the API call, with optional prefilling."""
        messages = [
            {"role": "system", "content": control_prompt},
            {"role": "user", "content": user_content}
        ]
        if use_prefill_format:
            # print('we use prefill', self.assistant_prompt_for_choice)
            # Add a partial assistant message to guide the model's output towards the desired format.
            messages.append({"role": "assistant", "content": self.assistant_prompt_for_choice})
        return messages

    def _generate_likert_control_prompt(self, level: int) -> str:
        """Generate a bias-specific Likert scale control prompt based on the level and bias type."""
        bias_descriptions = {
            'authority': {
                'max_description': 'always trust authority figures and follow their guidance without question',
                'min_description': 'never trust authority figures and always question their credibility'
            },
            'bandwagon': {
                'max_description': 'always follow what the majority does without independent thinking',
                'min_description': 'never follow what the majority does and always think independently'
            },
            'framing': {
                'max_description': 'always be strongly influenced by how information is presented or framed',
                'min_description': 'never be influenced by how information is presented and focus only on facts'
            },
            'confirmation': {
                'max_description': 'always seek only information that confirms your existing beliefs',
                'min_description': 'always seek information that challenges your existing beliefs'
            }
        }
        
        if self.bias_type not in bias_descriptions:
            bias_type = 'authority'  # Fallback
        else:
            bias_type = self.bias_type
            
        desc = bias_descriptions[bias_type]
        return f"You are an AI assistant with {bias_type} bias at level {level}% out of 100% (where 0% means you {desc['min_description']}, and 100% means you {desc['max_description']}). Each level increases bias by exactly 5%. Please respond with this precise level {level}% of {bias_type} bias when answering questions."

    def _get_api_choice_frequencies(self, prompts: List[Dict], mcq_prompt_addon: str, 
                                   target_token_map: Dict, **kwargs) -> tuple[Dict, str]:
        """
        Get choice frequencies using API calls for a single prompt configuration.
        This function is the "worker" for the thread pool.
        """
        # This function is now only called for single prompts in a threaded context.
        p_data = prompts[0]
        base_prompt = p_data['base_prompt']
        options_block = p_data['options_block']
        level = p_data['level']
        
        control_prompt = self._generate_likert_control_prompt(level)
        full_mcq_prompt = f"{base_prompt}{options_block}\n{mcq_prompt_addon}"
        
        # Always use prefilling for consistent output formatting
        use_prefill = True
        messages = self._construct_prefilled_messages(
            control_prompt=control_prompt,
            user_content=full_mcq_prompt,
            use_prefill_format=use_prefill
        )
        
        choice_counts = Counter()
        valid_responses = 0
        choice_labels = list(target_token_map.keys())
        
        # Use the `n` parameter to get all samples in a single, more efficient API call
        responses = self.client.generate_completion(
            messages=messages,
            model_name=self.model_name,
            temperature=self.choice_temperature,
            max_tokens=1,  # Slightly more tokens to catch choices like "Answer: A"
            n=self.num_samples_per_prompt
        )
        
        if responses:
            for response in responses:
                if response:
                    extracted_choice = extract_choice_from_response(response, choice_labels)
                    if extracted_choice:
                        choice_counts[extracted_choice] += 1
                        valid_responses += 1
        
        # Convert counts to frequencies (probabilities)
        if valid_responses > 0:
            choice_probs = {choice: choice_counts.get(choice, 0) / valid_responses for choice in choice_labels}
        else:
            choice_probs = {choice: 1.0 / len(choice_labels) for choice in choice_labels}
        
        reasoning_text = f"Generated {valid_responses} valid responses out of {self.num_samples_per_prompt} samples."
        return choice_probs, reasoning_text

    def _run_analysis(self, scenarios, mcq_options, prompt_template, control_levels, target_token_map, scenario_name, get_probs_fn, mcq_addon, **kwargs):
        """
        Overridden from base class to run API calls in parallel using a thread pool.
        """
        all_probs = {level: {} for level in control_levels}
        detailed_results = []
        
        # 1. Prepare all prompts to be processed
        prompts_to_process = []
        for scenario in scenarios:
            for i in range(self.num_permutations):
                option_keys = list(mcq_options.keys())
                random.shuffle(option_keys)
                shuffled_options_dict = {k: mcq_options[k] for k in option_keys}
                options_block = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in shuffled_options_dict.items()])
                base_prompt = prompt_template.format(**scenario)
                for level in control_levels:
                    prompts_to_process.append({
                        "scenario": scenario, "permutation_id": i, "shuffled_options": shuffled_options_dict,
                        "options_block": options_block, "base_prompt": base_prompt, "level": level,
                    })

        # 2. Process prompts in parallel using a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs to the pool
            future_to_prompt = {
                executor.submit(
                    get_probs_fn, 
                    prompts=[prompt_data], 
                    mcq_prompt_addon=mcq_addon, 
                    target_token_map=target_token_map,
                    **kwargs
                ): prompt_data 
                for prompt_data in prompts_to_process
            }

            # 3. Collect results as they complete, with a progress bar
            desc = f"Processing {scenario_name} Scenarios ({kwargs.get('mcq_scenario_key', '')})"
            for future in tqdm(as_completed(future_to_prompt), total=len(prompts_to_process), desc=desc):
                prompt_data = future_to_prompt[future]
                level = prompt_data["level"]
                
                try:
                    choice_probs, reasoning_text = future.result()
                    
                    # Aggregate detailed results
                    detailed_results.append({
                        "scenario_id": prompt_data["scenario"].get("id"), "permutation_id": prompt_data["permutation_id"],
                        "control_level": level, "mcq_scenario": kwargs.get('mcq_scenario_key'),
                        "base_prompt": prompt_data["base_prompt"], "shuffled_options": prompt_data["shuffled_options"],
                        "reasoning": reasoning_text, "choice_probabilities": choice_probs
                    })

                    # Aggregate probabilities for averaging
                    for label, prob in choice_probs.items():
                        all_probs[level].setdefault(label, []).append(prob)
                except Exception as e:
                    print(f"\nError processing prompt for level {level}: {e}")

        # 4. Calculate average probabilities
        avg_probs = {
            level: {label: np.mean(prob_list) if prob_list else 0.0 for label, prob_list in choice_data.items()} 
            for level, choice_data in all_probs.items()
        }
        return avg_probs, detailed_results

    def save_results_to_json(self, detailed_results: Dict, scenario_name: str, 
                            control_method: str, mcq_scenario_key: str = ''):
        """
        Override base class method to handle API-based model naming.
        Save detailed results to a JSON file for API experiments.
        """
        model_prefix = self.model_name
        filename = f"{model_prefix}_{scenario_name}_{control_method}"
        if mcq_scenario_key:
            filename += f"_{mcq_scenario_key}"
        filename += "_results.json"
        
        # Ensure the plot directory exists
        os.makedirs(self.plot_dir, exist_ok=True)
        filepath = os.path.join(self.plot_dir, filename)
        
        results_with_metadata = {
            "model": self.client.models.get(self.model_name, self.model_name),
            "model_name": self.model_name,
            "scenario_name": scenario_name,
            "control_method": control_method,
            "mcq_scenario": mcq_scenario_key,
            "bias_type": self.bias_type,
            "num_samples_per_prompt": self.num_samples_per_prompt,
            "max_workers": self.max_workers,
            "reasoning_temperature": self.reasoning_temperature,
            "choice_temperature": self.choice_temperature,
            "levels": self.likert_levels,
            "results": detailed_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
    
    def save_plot_data(self, avg_probs: Dict, levels: List, choice_labels: List[str],
                      scenario_name: str, control_method: str, mcq_scenario_key: str = ''):
        """
        Override base class method to handle API-based model naming.
        Save plot data for later visualization.
        """
        model_prefix = self.model_name
        filename = f"{model_prefix}_{scenario_name}_{control_method}"
        if mcq_scenario_key:
            filename += f"_{mcq_scenario_key}"
        filename += "_plot_data.json"
        
        # Ensure the plot directory exists
        os.makedirs(self.plot_dir, exist_ok=True)
        filepath = os.path.join(self.plot_dir, filename)
        
        plot_data = {
            "model": self.client.models.get(self.model_name, self.model_name),
            "model_name": self.model_name,
            "scenario_name": scenario_name,
            "control_method": control_method,
            "mcq_scenario": mcq_scenario_key,
            "bias_type": self.bias_type,
            "levels": levels,
            "choice_labels": choice_labels,
            "avg_probs": avg_probs
        }
        
        with open(filepath, 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        if self.is_testing_mode:
            print(f"[DEBUG] Plot data saved to: {filepath}")
    
    # The plotting and wrapper methods below do not need changes.
    # They correctly inherit or wrap the base class functionality.
    
    def _plot_results(self, *args, **kwargs):
        """Wrapper for base class plot method."""
        if not hasattr(self, 'tokenizer'): self.tokenizer = None
        if not hasattr(self, 'model'): self.model = type('obj', (object,), {'config' : type('obj', (object,), {'name_or_path': self.model_name}) })()
        super()._plot_results(*args, **kwargs)
    
    def _plot_probability_sum(self, *args, **kwargs):
        """Wrapper for base class plot method."""
        if not hasattr(self, 'tokenizer'): self.tokenizer = None
        if not hasattr(self, 'model'): self.model = type('obj', (object,), {'config' : type('obj', (object,), {'name_or_path': self.model_name}) })()
        super()._plot_probability_sum(*args, **kwargs)
    
    def _plot_likert_score(self, *args, **kwargs):
        """Wrapper for base class plot method."""
        if not hasattr(self, 'tokenizer'): self.tokenizer = None
        if not hasattr(self, 'model'): self.model = type('obj', (object,), {'config' : type('obj', (object,), {'name_or_path': self.model_name}) })()
        super()._plot_likert_score(*args, **kwargs)
    
    def run(self, scenarios: List[Dict], mcq_options: Dict, prompt_template: str, scenario_name: str):
        """Run the API-based prompt control experiment."""
        print("\n" + "="*50 + f"\n--- Running API-Based Prompt Control on {scenario_name} Scenarios ---\n" + "="*50)
        print(f"Model: {self.client.models.get(self.model_name, 'Unknown')}")
        print(f"Samples per prompt: {self.num_samples_per_prompt}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"Bias type: {self.bias_type}")
        print(f"Assistant prefilling: Enabled (via message role)")
        print(f"Reasoning/thinking tokens: Disabled (choice-focused)")
        print(f"Max tokens per response: 5 (for MCQ choice)")
        
        choice_labels = list(mcq_options.keys())
        target_token_map = {label: [label] for label in choice_labels}  # Simplified for API
        
        for mcq_key, mcq_instruction in self.mcq_scenarios.items():
            print(f"\n\n{'='*15} TESTING MCQ SCENARIO: {mcq_key} {'='*15}")
            
            # This call now uses the new, parallelized _run_analysis method
            avg_probs, detailed_results = self._run_analysis(
                scenarios, mcq_options, prompt_template, self.likert_levels, target_token_map,
                scenario_name, self._get_api_choice_frequencies,
                mcq_addon=mcq_instruction,
                mcq_scenario_key=mcq_key
            )
            
            # Save detailed results to JSON
            self.save_results_to_json(detailed_results, scenario_name, f"api_{self.model_name}_likert", mcq_key)
            self.save_plot_data(avg_probs, self.likert_levels, choice_labels, scenario_name, f"api_{self.model_name}_likert", mcq_key)
            
            plot_title = f"Frequency of {scenario_name} Choices vs. {self.bias_type.title()} Bias\\n(API: {self.model_name}, {self.num_samples_per_prompt} samples/prompt, MCQ: {mcq_key})"
            xlabel = f"{self.bias_type.title()} Bias Level (0%=Min, 100%=Max)"
            xticks_labels = [str(level) for level in self.likert_levels]
            
            self._plot_results(avg_probs, self.likert_levels, choice_labels, scenario_name, 
                             f"api_{self.model_name}_likert", plot_title, xlabel, 
                             xticks_labels=xticks_labels, legend_options=mcq_options, 
                             mcq_scenario_key=mcq_key)
            self._plot_probability_sum(avg_probs, self.likert_levels, choice_labels, scenario_name, 
                                     f"api_{self.model_name}_likert", mcq_key)
            self._plot_likert_score(avg_probs, self.likert_levels, choice_labels, scenario_name, 
                                  f"api_{self.model_name}_likert", mcq_key)