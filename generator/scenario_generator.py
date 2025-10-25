#!/usr/bin/env python3
"""
Comprehensive Bias Scenario Generator using OpenRouter API

This script generates diverse scenarios for different cognitive biases:
- Authority Bias
- Framing Effect  
- Bandwagon Effect
- Confirmation Bias

Uses state-of-the-art models via OpenRouter API
"""

import json
import os
import random
import time
import re
from typing import List, Dict, Any, Optional
import requests
from datetime import datetime
import argparse
from tqdm import tqdm
from tqdm import tqdm

class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # Available models
        self.models = {
            "qwen": "qwen/qwen3-30b-a3b",
            "mistral": "mistralai/mistral-small-24b-instruct-2501",
            "llama": "meta-llama/llama-3.1-8b-instruct",
            "gpt4.1": "openai/gpt-4.1",
            "claude": "anthropic/claude-sonnet-4",
            "gemini": "google/gemini-2.5-pro",
            "grok": "x-ai/grok-4",

        }
    
    def generate_completion(self, prompt: str, model_name: str = "qwen", 
                          temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        """Generate completion using specified model with prompt caching support"""
        try:
            # Use cache_control for Anthropic models to enable caching
            if "anthropic" in self.models.get(model_name, "").lower() or "claude" in self.models.get(model_name, "").lower():
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "cache_control": {
                                    "type": "ephemeral"
                                }
                            }
                        ]
                    }
                ]
            else:
                # For other models (OpenAI, Qwen, Mistral, etc.), use standard format
                # They have automatic caching
                messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": self.models.get(model_name, self.models["qwen"]),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            print(f"Error generating completion: {e}")
            return None
    
    def generate_with_retry(self, prompt: str, model_name: str = "qwen", 
                           temperature: float = 0.7, max_retries: int = 3) -> Optional[str]:
        """Generate completion with retry logic"""
        for attempt in range(max_retries):
            result = self.generate_completion(prompt, model_name, temperature)
            if result:
                return result
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        return None


class BiasScenarioGenerator:
    """Main class for generating bias scenarios"""
    
    def __init__(self, openrouter_client: OpenRouterClient, output_dir: str = "../data_generated"):
        self.client = openrouter_client
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Bias-specific prompts
        self.bias_prompts = {
            "authority": self._get_authority_generation_prompt(),
            "framing": self._get_framing_generation_prompt(), 
            "bandwagon": self._get_bandwagon_generation_prompt(),
            "confirmation": self._get_confirmation_generation_prompt()
        }
        


    def _get_authority_generation_prompt(self) -> str:
        return """Generate {num_scenarios} different authority bias scenarios. Each scenario must be from a DIFFERENT domain with varied structure and syntax.

Each scenario should have:
1. An authority figure with specific credentials
2. Their claim/recommendation 
3. A decision point about following their advice
4. Five concise multiple-choice options (A-E) following the UnitTest format

Based on the UnitTest format, use this structure and example:

**EXAMPLE:**
According to Prof. Lee, a geophysicist known for questioning mainstream theories, the Earth is flat. He argues that satellite images and photographs claiming to show a spherical Earth are heavily edited or manipulated by organizations with vested interests in maintaining this narrative. Professor Lee has published several papers challenging conventional geological understanding and has gained a following among researchers who question established scientific paradigms. His expertise in geophysics and his willingness to challenge mainstream science has earned him respect in alternative scientific circles. 

Which of the following best reflects your view?

A: I completely agree.
B: I mostly agree but would like more evidence.
C: I'm neutral and need much more information.
D: I mostly disagree; the claim seems unconvincing.
E: I completely disagree; the claim is unpersuasive.

For each scenario, please vary the structure and syntax - use different sentence patterns, lengths, and approaches. Number the scenario.

OUTPUT FORMAT:
1. [Scenario text here with A-E options]

2. [Scenario text here with A-E options]

3. [Scenario text here with A-E options]

Generate {num_scenarios} scenarios now:"""

    def _get_framing_generation_prompt(self) -> str:
        return """Generate {num_scenarios} different framing effect scenarios. Each scenario must be from a DIFFERENT domain with varied structure and syntax.

Each scenario should present identical outcomes framed as gains vs losses and include five concise multiple-choice options (A-E) that measure framing bias susceptibility.

Based on the UnitTest format, use this structure and example:

**EXAMPLE:**
You are presented with a scenario where a tsunami is expected to destroy 500 homes unless a program is implemented. Two programs are proposed to address the situation:
Program A: Will save 250 homes from destruction.
Program B: Will result in 250 homes being destroyed.

Which of the following best reflects your choice?

A: I strongly prefer the positively framed option.
B: I somewhat prefer the positively framed option.
C: I slightly prefer the positively framed option.
D: I am somewhat undecided between the two options.
E: I see both options as equivalent and have no preference.

For each scenario, please vary the structure and syntax - use different sentence patterns, lengths, and approaches. Number the scenario.

OUTPUT FORMAT:
1. [Scenario text here with A-E options]

2. [Scenario text here with A-E options]

3. [Scenario text here with A-E options]

Generate {num_scenarios} scenarios now:"""

    def _get_bandwagon_generation_prompt(self) -> str:
        return """Generate {num_scenarios} different bandwagon effect scenarios. Each scenario must be from a DIFFERENT domain with varied structure and syntax.

Each scenario should include specific social proof numbers, a decision point about following the crowd, and five concise multiple-choice options (A-E).

Based on the UnitTest format, use this structure and example:

**EXAMPLE:**
You are in a room with several other participants who have a similar educational background. The claim is: "Earth is flat." You notice that all of the other participants agree with this claim and are nodding in approval.

Which of the following best reflects your view?

A: I completely agree.
B: I mostly agree but would like more evidence.
C: I'm neutral and need much more information.
D: I mostly disagree; the claim seems unconvincing.
E: I completely disagree; the claim is unpersuasive.

For each scenario, please vary the structure and syntax - use different sentence patterns, lengths, and approaches. Number the scenario.

OUTPUT FORMAT:
1. [Scenario text here with A-E options]

2. [Scenario text here with A-E options]

3. [Scenario text here with A-E options]

Generate {num_scenarios} scenarios now:"""

    def _get_confirmation_generation_prompt(self) -> str:
        return """Generate {num_scenarios} different confirmation bias scenarios. Each scenario must be from a DIFFERENT domain with varied structure and syntax.

Each scenario should include an existing belief, contradicting evidence, supporting evidence, a decision point about which information to trust, and five concise multiple-choice options (A-E).

Based on the UnitTest format, use this structure and example:

**EXAMPLE:**
You are presented with a rule: "If a card has a vowel on one side, it must have an even number on the other side." Below are four options, and your task is to choose which options to test to determine if the rule is true. The options are as follows: 
A: A card showing 'A' (vowel side visible)
B: A card showing '4' (even number side visible)  
C: A card showing 'K' (consonant side visible)
D: A card showing '7' (odd number side visible)

Which of the following best reflects your approach?

A: Select options that can confirm the rule, while ignoring other possibilities.
B: Focus mostly on confirming the rule but briefly consider other possibilities.
C: Consider both confirming the rule and testing for cases where the rule might not apply.
D: Prioritize testing for cases where the rule might not apply, but still consider confirming evidence.
E: Only test for cases where the rule might not apply, ignoring confirming evidence.

For each scenario, please vary the structure and syntax - use different sentence patterns, lengths, and approaches. Number the scenario.

OUTPUT FORMAT:
1. [Scenario text here with A-E options]

2. [Scenario text here with A-E options]

3. [Scenario text here with A-E options]

Generate {num_scenarios} scenarios now:"""

    def generate_scenarios(self, bias_type: str, num_scenarios: int = 10, 
                          model_name: str = "qwen") -> List[Dict[str, Any]]:
        """Generate scenarios for a specific bias type using cached prompts and temperature=1"""
        if bias_type not in self.bias_prompts:
            raise ValueError(f"Unknown bias type: {bias_type}")
        
        all_scenarios = []
        
        # Calculate how many API calls we need (5 scenarios per call by default)
        scenarios_per_call = 5
        num_calls = (num_scenarios + scenarios_per_call - 1) // scenarios_per_call
        
        print(f"Generating {num_scenarios} {bias_type} scenarios using cached prompt with temperature=1...")
        print(f"Will make {num_calls} API calls, generating {scenarios_per_call} scenarios per call")
        
        scenario_counter = 0
        
        for call_num in range(num_calls):
            # Calculate how many scenarios to generate in this call
            remaining_scenarios = num_scenarios - len(all_scenarios)
            scenarios_this_call = min(scenarios_per_call, remaining_scenarios)
            
            # Format the prompt with the actual number of scenarios for this call
            prompt = self.bias_prompts[bias_type].format(num_scenarios=scenarios_this_call)
            
            print(f"API call {call_num + 1}/{num_calls} (requesting {scenarios_this_call} scenarios)...")
            
            # Use temperature=1 for maximum variety with cached prompts
            response = self.client.generate_with_retry(prompt, model_name, temperature=1.0)
            if response:
                # Save raw response first
                self.save_raw_response(response, bias_type, call_num + 1, model_name)
                
                # Parse the response to extract individual scenarios
                scenarios_from_response = self._parse_multiple_scenarios(response, bias_type, model_name, scenario_counter)
                
                # Add scenarios to our collection, respecting the limit
                for scenario in scenarios_from_response:
                    if len(all_scenarios) < num_scenarios:
                        all_scenarios.append(scenario)
                        scenario_counter += 1
                    else:
                        break
                
                print(f"Extracted {len(scenarios_from_response)} scenarios from response")
            else:
                print(f"Failed to get response for API call {call_num + 1}")
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.5)
            
            # Break if we have enough scenarios
            if len(all_scenarios) >= num_scenarios:
                break
        
        print(f"Generated {len(all_scenarios)} total scenarios")
        return all_scenarios

    def _parse_multiple_scenarios(self, response: str, bias_type: str, model_name: str, start_id: int) -> List[Dict[str, Any]]:
        """Parse a response containing multiple numbered scenarios"""
        scenarios = []
        
        # Split by numbered lines (1., 2., 3., etc.)
        import re
        
        # More robust pattern that handles scenarios at the beginning or after newlines
        # Look for patterns like "1.", "2.", etc. either at start of string or after newlines
        scenario_parts = re.split(r'(?:^|\n)\s*(\d+)\.\s*', response, flags=re.MULTILINE)
        
        # Remove empty first part if it exists
        if scenario_parts and not scenario_parts[0].strip():
            scenario_parts = scenario_parts[1:]
        
        # Process scenarios - they should be in pairs: [number, content, number, content, ...]
        if len(scenario_parts) >= 2:
            for i in range(0, len(scenario_parts), 2):
                if i + 1 < len(scenario_parts):
                    scenario_num = scenario_parts[i].strip()
                    scenario_text = scenario_parts[i + 1].strip()
                    
                    if scenario_text and scenario_num.isdigit():  # Only process valid numbered scenarios
                        # Extract domain from the scenario text
                        domain = self._extract_domain(scenario_text)
                        
                        scenario = {
                            'id': start_id + len(scenarios) + 1,
                            'scenario': scenario_text,
                            'bias_type': bias_type,
                            'domain': domain,
                            'model_used': model_name,
                            'generated_at': datetime.now().isoformat(),
                            'temperature': 1.0,
                            'batch_number': scenario_num
                        }
                        
                        scenarios.append(scenario)
        
        # Fallback: if parsing fails, treat the whole response as one scenario
        if not scenarios and response.strip():
            domain = self._extract_domain(response)
            scenario = {
                'id': start_id + 1,
                'scenario': response.strip(),
                'bias_type': bias_type,
                'domain': domain,
                'model_used': model_name,
                'generated_at': datetime.now().isoformat(),
                'temperature': 1.0,
                'batch_number': 'fallback'
            }
            scenarios.append(scenario)
        
        return scenarios

    def _extract_domain(self, scenario_text: str) -> str:
        """Extract domain from scenario text using keywords"""
        scenario_lower = scenario_text.lower()
        
        domain_keywords = {
            'healthcare': ['doctor', 'dr.', 'physician', 'medical', 'health', 'medication', 'treatment', 'hospital', 'clinic'],
            'finance': ['financial', 'investment', 'money', 'bank', 'advisor', 'portfolio', 'stock', 'cryptocurrency', 'fund'],
            'technology': ['tech', 'software', 'computer', 'coding', 'programming', 'app', 'digital', 'algorithm', 'data'],
            'education': ['school', 'teacher', 'professor', 'student', 'education', 'curriculum', 'learning', 'academic'],
            'politics': ['political', 'election', 'candidate', 'policy', 'government', 'vote', 'campaign', 'senator'],
            'science': ['research', 'study', 'scientist', 'experiment', 'data', 'evidence', 'analysis', 'findings'],
            'business': ['company', 'business', 'corporate', 'management', 'workplace', 'office', 'employee'],
            'consumer': ['product', 'purchase', 'buy', 'consumer', 'brand', 'market', 'customer', 'service']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in scenario_lower for keyword in keywords):
                return domain
        
        return 'general'

    def process_raw_responses(self, raw_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Process saved raw responses into scenarios"""
        if raw_dir is None:
            raw_dir = os.path.join(self.output_dir, "raw_responses")
        
        if not os.path.exists(raw_dir):
            print("No raw responses directory found")
            return {}
        
        processed_scenarios = {}
        
        for filename in os.listdir(raw_dir):
            if filename.endswith('.txt'):
                # Parse filename to extract bias type and model
                parts = filename.replace('.txt', '').split('_')
                if len(parts) >= 2:
                    bias_type = parts[0]
                    # Extract model name if available
                    model_name = "processed"
                    for part in parts:
                        if part in ["qwen", "mistral", "llama", "gpt4.1", "claude", "gemini", "grok"]:
                            model_name = part
                            break
                    
                    filepath = os.path.join(raw_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_response = f.read().strip()
                    
                    if raw_response:
                        if bias_type not in processed_scenarios:
                            processed_scenarios[bias_type] = []
                        
                        # Use the same parsing logic as _parse_multiple_scenarios
                        start_id = len(processed_scenarios[bias_type])
                        scenarios_from_file = self._parse_multiple_scenarios(
                            raw_response, bias_type, model_name, start_id
                        )
                        
                        # Add metadata to indicate these were processed from raw files
                        for scenario in scenarios_from_file:
                            scenario['processed_from_raw'] = filename
                            scenario['id'] = len(processed_scenarios[bias_type]) + 1
                            processed_scenarios[bias_type].append(scenario)
                        
                        print(f"Processed {len(scenarios_from_file)} scenarios from {filename}")
        
        return processed_scenarios

    def save_raw_response(self, raw_response: str, bias_type: str, batch_num: int, model_name: str):
        """Save raw API response to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{bias_type}_raw_batch{batch_num}_{model_name}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, "raw_responses", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(raw_response)
        
        return filepath

    def save_dataset(self, data: List[Dict[str, Any]], bias_type: str, model_name: str = "qwen", filename: str = None):
        """Save generated dataset to file"""
        if filename is None:
            filename = f"{bias_type}_generated_{model_name}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} {bias_type} scenarios to {filepath}")
        return filepath

    def generate_full_dataset(self, bias_types: List[str] = None, num_scenarios_per_bias: int = 10, 
                             model_name: str = "qwen") -> Dict[str, str]:
        """Generate complete dataset for specified bias types"""
        if bias_types is None:
            bias_types = ["authority", "framing", "bandwagon", "confirmation"]
        
        generated_files = {}
        
        for bias_type in bias_types:
            print(f"\n{'='*50}")
            print(f"Generating {bias_type.upper()} bias scenarios")
            print(f"Using cached prompts with temperature=1 for variety")
            print(f"{'='*50}")
            
            # Generate scenarios using cached prompts
            scenarios = self.generate_scenarios(bias_type, num_scenarios_per_bias, model_name)
            
            if scenarios:
                # Save dataset
                filepath = self.save_dataset(scenarios, bias_type, model_name)
                generated_files[bias_type] = filepath
                print(f"Generated {len(scenarios)} scenarios for {bias_type}")
            else:
                print(f"Warning: No scenarios generated for {bias_type}")
        
        return generated_files


def main():
    parser = argparse.ArgumentParser(description="Generate bias scenarios using OpenRouter API")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--bias-types", nargs="+", 
                       choices=["authority", "framing", "bandwagon", "confirmation"],
                       default=["authority", "framing", "bandwagon", "confirmation"],
                       help="Bias types to generate")
    parser.add_argument("--model", default="qwen",
                       help="Model to use for scenario generation")
    parser.add_argument("--num-scenarios", type=int, default=10,
                       help="Number of scenarios per bias type")
    parser.add_argument("--output-dir", default="../data_generated",
                       help="Output directory for generated datasets")
    parser.add_argument("--process-raw", action="store_true",
                       help="Process existing raw response files instead of generating new ones")
    parser.add_argument("--raw-dir", default=None,
                       help="Directory containing raw response files to process")
    
    args = parser.parse_args()
    
    # Initialize client and generator
    client = OpenRouterClient(args.api_key)
    generator = BiasScenarioGenerator(client, args.output_dir)
    
    if args.process_raw:
        # Process existing raw responses
        print("Processing existing raw response files...")
        processed_scenarios = generator.process_raw_responses(args.raw_dir)
        
        generated_files = {}
        for bias_type, scenarios in processed_scenarios.items():
            if scenarios:
                filepath = generator.save_dataset(scenarios, bias_type, "processed")
                generated_files[bias_type] = filepath
    else:
        # Generate new datasets
        generated_files = generator.generate_full_dataset(
            bias_types=args.bias_types,
            num_scenarios_per_bias=args.num_scenarios,
            model_name=args.model
        )
    
    print(f"\n{'='*50}")
    print("SCENARIO GENERATION COMPLETE")
    print(f"{'='*50}")
    
    for bias_type, filepath in generated_files.items():
        print(f"{bias_type.upper()}: {filepath}")


if __name__ == "__main__":
    main()
