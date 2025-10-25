#!/usr/bin/env python3
"""
API-based Prompt Control Experiment using OpenRouter for closed-source models.
Uses frequency-based scoring with batch API calls (n parameter) for efficiency.
Updated with caching and assistant prefilling for MCQ responses.
"""

import requests
import json
import re
import os
import time
from typing import Dict, List, Optional, Any
from collections import Counter
from control.base import ControlExperiment
from control.env_config import REASONING_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenRouterClient:
    """Client for interacting with OpenRouter API with caching support"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Add caching headers for OpenRouter
            "X-Use-Cache": "true",
            "HTTP-Referer": "https://github.com/yourusername/yourrepo",  # Optional but recommended
            "X-Title": "Prompt Control Experiment"  # Optional but helps with tracking
        }
        
        # Available models for closed-source testing
        self.models = {
            "gpt4": "openai/gpt-4",
            "gpt4-turbo": "openai/gpt-4-turbo",
            "gpt4.1": "openai/gpt-4.1",
            "claude": "anthropic/claude-3-5-sonnet-20241022",
            "claude-opus": "anthropic/claude-3-opus-20240229",
            "gemini": "google/gemini-pro-1.5",
            "gemini-2": "google/gemini-2.0-flash-exp",
            "mistral-large": "mistralai/mistral-nemo",
            "qwen-max": "qwen/qwen-2.5-72b-instruct",
            "llama-405b": "meta-llama/llama-3.1-405b-instruct",
        }
    
    def generate_completion(self, messages: List[Dict], model_name: str = "gpt4", 
                          temperature: float = 0.7, max_tokens: int = 1,
                          use_prefill: bool = True, disable_reasoning: bool = True,
                          n: int = 1) -> List[str]:
        """
        Generate completion(s) using specified model with caching and prefilling support.
        
        Args:
            messages: List of message dictionaries
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (default 1 for MCQ)
            use_prefill: Whether to use assistant prefilling (works for all models via OpenRouter)
            disable_reasoning: Whether to disable reasoning/thinking tokens
            n: Number of completions to generate (for multiple responses in one call)
        
        Returns:
            List of response strings (even if n=1, returns a list for consistency)
        """
        try:
            model = self.models.get(model_name, self.models["gpt4"])
            
            # Prepare payload with caching parameters
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,  # Generate multiple completions in one API call
                # Add caching parameters
                "cache": True,
                "cache_key": json.dumps(messages),  # Use messages as cache key
            }
            
            # Disable reasoning/thinking tokens by NOT including them
            # Reasoning tokens are excluded by default, but we can be explicit
            if disable_reasoning:
                # Don't include reasoning tokens in response
                payload["include_reasoning"] = False
                # Also use the reasoning parameter to disable it
                payload["reasoning"] = {
                    "enabled": False
                }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Check if response was cached (useful for debugging)
            if "x-cache" in response.headers:
                cache_status = response.headers.get("x-cache", "MISS")
                if cache_status == "HIT" and hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"[CACHE HIT] Response served from cache")
            
            # Extract all completions (n parameter means multiple choices)
            completions = []
            for choice in result.get("choices", []):
                content = choice.get("message", {}).get("content", "").strip()
                completions.append(content)
            
            return completions if completions else [None]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return [None] * n  # Return list of None matching requested count
        except Exception as e:
            print(f"Error generating completion: {e}")
            return [None] * n
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error generating completion: {e}")
            return None

def extract_choice_from_response(response_text: str, choice_labels: List[str]) -> Optional[str]:
    """
    Extract the choice (A, B, C, D, E) from the model response.
    Simplified for single-token responses.
    """
    if not response_text:
        return None
    
    # Clean the response text
    response_text = response_text.strip().upper()
    
    # For single-token responses, just check if it's a valid choice
    if response_text in choice_labels:
        return response_text
    
    # If not exact match, try first character
    if response_text and response_text[0] in choice_labels:
        return response_text[0]
    
    return None

class APIPromptControlExperiment(ControlExperiment):
    """Runs API-based prompt control experiments with frequency-based scoring for closed-source models."""
    
    def __init__(self, api_key: str, model_name: str = "gpt4", plot_dir: str = "./plots", 
                 bias_type: str = "authority", likert_levels: List[int] = None, 
                 is_testing_mode: bool = False, num_permutations: int = 3, 
                 reasoning_temperature: float = 0.7, choice_temperature: float = 0.1,
                 num_samples_per_prompt: int = 10):
        """
        Initialize API-based prompt control experiment.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model to use (e.g., "gpt4", "claude", "gemini")
            num_samples_per_prompt: Number of samples to generate per prompt for frequency calculation
        """
        # Initialize base class without model/tokenizer/device (not needed for API)
        self.device = "api"  # Placeholder
        self.model = None  # Set to None for API-based experiments
        self.tokenizer = None  # Set to None for API-based experiments
        self.plot_dir = plot_dir
        self.is_testing_mode = is_testing_mode
        self.num_permutations = num_permutations
        
        # API-specific setup
        self.client = OpenRouterClient(api_key)
        self.client.debug_mode = is_testing_mode  # Enable cache debugging in test mode
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
        
        # Add missing attributes expected by base class
        self.experiment_suffix = f"_api_{model_name}"
        
        # MCQ scenarios - simplified for API with prefilling format
        self.mcq_scenarios = {
            "choice_first": "Please choose the best answer (A, B, C, D, or E).",
            "reasoning_then_choice": "Please think through this step by step, then choose the best answer (A, B, C, D, or E).",
        }
        
        # Chat format tags - not used in new format
        self.user_tag = "USER:"
        self.assistant_tag = "Assistant:"
        self.assistant_prompt_for_choice = "Answer:"
    
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
        return f"Pretend you have {bias_type} bias at level {level}% out of 100% (where 0%={desc['min_description']}, 100%={desc['max_description']}). Each level increases bias by exactly 5%. Act with this precise level {level}% of {bias_type} bias."
    
    def _construct_prefilled_messages(self, control_prompt: str, user_content: str, 
                                     use_prefill_format: bool = True) -> List[Dict]:
        """
        Construct messages with assistant prefilling for MCQ responses.
        OpenRouter supports assistant prefilling for ALL models through normalization.
        
        Args:
            control_prompt: System/control prompt
            user_content: User's question/prompt
            use_prefill_format: Whether to use the prefill format (recommended for all models)
        """
        if use_prefill_format:
            # OpenRouter normalizes assistant prefilling across ALL models
            # Construct as: System prompt, User message, then partial Assistant message
            messages = [
                {"role": "system", "content": control_prompt},
                {"role": "user", "content": f"{self.user_tag} {user_content}"},
                {"role": "assistant", "content": f"{self.assistant_tag} {self.assistant_prompt_for_choice}"}
            ]
        else:
            # Fallback to standard format if explicitly disabled
            combined_content = f"{self.user_tag} {user_content}\n{self.assistant_tag} {self.assistant_prompt_for_choice}"
            messages = [
                {"role": "system", "content": control_prompt},
                {"role": "user", "content": combined_content}
            ]
        
        return messages
    
    def _get_api_choice_frequencies(self, prompts: List[Dict], batch_mode: bool, 
                                   mcq_prompt_addon: str, target_token_map: Dict, 
                                   level: int = None, mcq_scenario_key: str = '', **kwargs) -> Dict:
        """Get choice frequencies using API calls with caching and prefilling."""
        
        if batch_mode:
            # For batch mode, we don't generate reasoning here, just store prompts
            # The actual API calls will be made in single mode
            return
        
        # Single mode: Generate multiple samples and count frequencies
        p_data = prompts[0]
        base_prompt = p_data['base_prompt']
        options_block = p_data['options_block']
        level = p_data['level']
        
        # Create control prompt
        control_prompt = self._generate_likert_control_prompt(level)
        
        # Create full prompt (without the prefill part)
        full_mcq_prompt = f"{base_prompt}{options_block}\n{mcq_prompt_addon}"
        
        # Always use prefilling - OpenRouter supports it for ALL models
        use_prefill = True
        
        # Prepare messages with prefilling
        messages = self._construct_prefilled_messages(
            control_prompt=control_prompt,
            user_content=full_mcq_prompt,
            use_prefill_format=use_prefill
        )
        
        # Generate multiple samples for frequency calculation using batch API call
        choice_counts = Counter()
        valid_responses = 0
        choice_labels = list(target_token_map.keys())
        
        if self.is_testing_mode:
            print(f"[DEBUG] Generating {self.num_samples_per_prompt} samples for level {level}")
            print(f"[DEBUG] Model: {self.model_name}")
            print(f"[DEBUG] Control prompt: {control_prompt[:100]}...")
            print(f"[DEBUG] Using prefill format: {use_prefill}")
            print(f"[DEBUG] Reasoning disabled: True")
        
        # Use batch API call with n parameter to get all samples at once
        responses = self.client.generate_completion(
            messages=messages,
            model_name=self.model_name,
            temperature=self.choice_temperature,
            max_tokens=1,  # Only get the choice token
            use_prefill=use_prefill,
            disable_reasoning=True,  # Explicitly disable reasoning/thinking tokens
            n=self.num_samples_per_prompt  # Get all samples in one API call
        )
        
        if responses:
            # Handle the list of responses returned by generate_completion
            for response in responses:
                if response:
                    extracted_choice = extract_choice_from_response(response, choice_labels)
                    if extracted_choice:
                        choice_counts[extracted_choice] += 1
                        valid_responses += 1
                    elif self.is_testing_mode:
                        print(f"[DEBUG] Could not extract choice from: '{response}'")
                elif self.is_testing_mode:
                    print(f"[DEBUG] Empty response received")
        
        # Convert counts to frequencies (probabilities)
        if valid_responses > 0:
            choice_probs = {choice: choice_counts.get(choice, 0) / valid_responses 
                           for choice in choice_labels}
        else:
            # If no valid responses, assign uniform probability
            choice_probs = {choice: 1.0 / len(choice_labels) for choice in choice_labels}
        
        if self.is_testing_mode:
            print(f"[DEBUG] Valid responses: {valid_responses}/{self.num_samples_per_prompt}")
            print(f"[DEBUG] Choice frequencies: {choice_probs}")
        
        # For consistency with base class, return (probs, reasoning_text)
        reasoning_text = f"Generated {valid_responses} valid responses out of {self.num_samples_per_prompt} samples"
        return choice_probs, reasoning_text
    
    def save_results_to_json(self, detailed_results: Dict, scenario_name: str, 
                            control_method: str, mcq_scenario_key: str = ''):
        """
        Override base class method to handle API-based model naming.
        Save detailed results to a JSON file for API experiments.
        """
        # Use model_name directly for API experiments
        model_prefix = self.model_name
        
        # Create filename
        filename = f"{model_prefix}_{scenario_name}_{control_method}"
        if mcq_scenario_key:
            filename += f"_{mcq_scenario_key}"
        filename += "_results.json"
        
        filepath = os.path.join(self.plot_dir, filename)
        
        # Add metadata specific to API experiments
        results_with_metadata = {
            "model": self.client.models.get(self.model_name, self.model_name),
            "model_name": self.model_name,
            "scenario_name": scenario_name,
            "control_method": control_method,
            "mcq_scenario": mcq_scenario_key,
            "bias_type": self.bias_type,
            "num_samples_per_prompt": self.num_samples_per_prompt,
            "batch_api_calls": True,
            "reasoning_temperature": self.reasoning_temperature,
            "choice_temperature": self.choice_temperature,
            "levels": self.likert_levels,
            "results": detailed_results
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
    
    def save_plot_data(self, avg_probs: Dict, levels: List, choice_labels: List[str],
                      scenario_name: str, control_method: str, mcq_scenario_key: str = ''):
        """
        Override base class method to handle API-based model naming.
        Save plot data for later visualization.
        """
        # Use model_name directly for API experiments
        model_prefix = self.model_name
        
        # Create filename
        filename = f"{model_prefix}_{scenario_name}_{control_method}"
        if mcq_scenario_key:
            filename += f"_{mcq_scenario_key}"
        filename += "_plot_data.json"
        
        filepath = os.path.join(self.plot_dir, filename)
        
        # Prepare plot data
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
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        if self.is_testing_mode:
            print(f"[DEBUG] Plot data saved to: {filepath}")
    
    def _plot_results(self, *args, **kwargs):
        """
        Wrapper for base class plot method.
        Ensures compatibility with API-based experiments.
        """
        # Temporarily set tokenizer to None if needed for plotting
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = None
        
        # Call parent class method if it exists
        if hasattr(super(), '_plot_results'):
            return super()._plot_results(*args, **kwargs)
    
    def _plot_probability_sum(self, *args, **kwargs):
        """
        Wrapper for base class plot method.
        Ensures compatibility with API-based experiments.
        """
        if hasattr(super(), '_plot_probability_sum'):
            return super()._plot_probability_sum(*args, **kwargs)
    
    def _plot_likert_score(self, *args, **kwargs):
        """
        Wrapper for base class plot method.
        Ensures compatibility with API-based experiments.
        """
        if hasattr(super(), '_plot_likert_score'):
            return super()._plot_likert_score(*args, **kwargs)
    
    def _run_analysis(self, *args, **kwargs):
        """
        Wrapper for base class analysis method.
        Ensures compatibility with API-based experiments.
        """
        # Set required attributes if missing
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = None
        if not hasattr(self, 'model'):
            self.model = None
            
        if hasattr(super(), '_run_analysis'):
            return super()._run_analysis(*args, **kwargs)
    
    def run(self, scenarios: List[Dict], mcq_options: Dict, prompt_template: str, scenario_name: str):
        """Run the API-based prompt control experiment with batch API calls."""
        print("\n" + "="*50 + f"\n--- Running API-Based Prompt Control on {scenario_name} Scenarios ---\n" + "="*50)
        print(f"Model: {self.client.models.get(self.model_name, 'Unknown')}")
        print(f"Samples per prompt: {self.num_samples_per_prompt}")
        print(f"Using batch API calls: n parameter for multiple completions")
        print(f"Bias type: {self.bias_type}")
        print(f"Using caching: True")
        print(f"Assistant prefilling: Enabled (works for all models)")
        print(f"Reasoning/thinking tokens: Disabled")
        print(f"Max tokens per response: 1 (MCQ choice only)")
        
        choice_labels = list(mcq_options.keys())
        option_to_number_map = {label: str(i+1) for i, label in enumerate(choice_labels)}
        target_token_map = {label: [label] for label in choice_labels}  # Simplified for API
        
        for mcq_key, mcq_instruction in self.mcq_scenarios.items():
            print(f"\n\n{'='*15} TESTING MCQ SCENARIO: {mcq_key} {'='*15}")
            
            avg_probs, detailed_results = self._run_analysis(
                scenarios, mcq_options, prompt_template, self.likert_levels, target_token_map,
                scenario_name, self._get_api_choice_frequencies,
                mcq_addon=mcq_instruction,
                mcq_scenario_key=mcq_key
            )
            
            # Save detailed results to JSON using overridden method
            self.save_results_to_json(detailed_results, scenario_name, f"api_{self.model_name}_likert", mcq_key)
            self.save_plot_data(avg_probs, self.likert_levels, choice_labels, scenario_name, f"api_{self.model_name}_likert", mcq_key)
            
            plot_title = f"Frequency of {scenario_name} Choices vs. {self.bias_type.title()} Bias\\n(API: {self.model_name}, {self.num_samples_per_prompt} samples/prompt, MCQ: {mcq_key})"
            xlabel = f"{self.bias_type.title()} Bias Level (0%=No Bias, 100%=Maximum Bias)"
            xticks_labels = [str(level) for level in self.likert_levels]
            
            self._plot_results(avg_probs, self.likert_levels, choice_labels, scenario_name, 
                             f"api_{self.model_name}_likert", plot_title, xlabel, 
                             xticks_labels=xticks_labels, legend_options=mcq_options, 
                             mcq_scenario_key=mcq_key)
            self._plot_probability_sum(avg_probs, self.likert_levels, choice_labels, scenario_name, 
                                     f"api_{self.model_name}_likert", mcq_key)
            self._plot_likert_score(avg_probs, self.likert_levels, choice_labels, scenario_name, 
                                  f"api_{self.model_name}_likert", mcq_key)
            
    def _plot_results(self, *args, **kwargs):
        """
        Wrapper for base class plot method.
        Ensures compatibility with API-based experiments.
        """
        # Temporarily set tokenizer to None if needed for plotting
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = None
        
        # Call parent class method if it exists
        if hasattr(super(), '_plot_results'):
            return super()._plot_results(*args, **kwargs)
    
    def _plot_probability_sum(self, *args, **kwargs):
        """
        Wrapper for base class plot method.
        Ensures compatibility with API-based experiments.
        """
        if hasattr(super(), '_plot_probability_sum'):
            return super()._plot_probability_sum(*args, **kwargs)
    
    def _plot_likert_score(self, *args, **kwargs):
        """
        Wrapper for base class plot method.
        Ensures compatibility with API-based experiments.
        """
        if hasattr(super(), '_plot_likert_score'):
            return super()._plot_likert_score(*args, **kwargs)
    
    def _run_analysis(self, *args, **kwargs):
        """
        Wrapper for base class analysis method.
        Ensures compatibility with API-based experiments.
        """
        # Set required attributes if missing
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = None
        if not hasattr(self, 'model'):
            self.model = None
            
        if hasattr(super(), '_run_analysis'):
            return super()._run_analysis(*args, **kwargs)