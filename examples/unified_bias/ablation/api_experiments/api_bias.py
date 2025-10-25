#!/usr/bin/env python3
"""
API-based bias experiment runner for closed-source models via OpenRouter.
Uses frequency-based scoring instead of probability-based scoring.
"""

import argparse
import os
import sys
import time

# Add control directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from control.api_prompt_experiment import APIPromptControlExperiment
from control.env_config import OPENROUTER_API_KEY, print_config

# Add unified_bias utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils_bias import load_scenarios_and_options

def main():
    parser = argparse.ArgumentParser(description='Run API-based bias experiments using OpenRouter')
    parser.add_argument('--bias', choices=['authority', 'bandwagon', 'framing', 'confirmation', 'all'], 
                       required=True, help='Type of bias to test')
    parser.add_argument('--model', default='gpt4', help='API model to use')
    parser.add_argument('--api-key', type=str, help='OpenRouter API key (overrides .env file)')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of samples per prompt for frequency calculation')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Enable debug output for testing')
    parser.add_argument('--permutations', type=int, default=3, 
                       help='Number of permutations to run')
    parser.add_argument('--reasoning-temp', type=float, default=0.7, 
                       help='Temperature for reasoning generation')
    parser.add_argument('--choice-temp', type=float, default=0.7, 
                       help='Temperature for choice selection')
    parser.add_argument('--levels', type=str, help='Comma-separated list of bias levels (e.g., "0,25,50,75,100")')
    parser.add_argument('--max-workers', type=int, default=5, 
                       help='Maximum number of concurrent API requests')
    
    args = parser.parse_args()
    
    # Print configuration
    print_config()
    
    # Get API key
    api_key = args.api_key or OPENROUTER_API_KEY
    if not api_key:
        print("ERROR: OpenRouter API key not found!")
        print("Please either:")
        print("1. Set OPENROUTER_API_KEY in your .env file, or")
        print("2. Use --api-key argument")
        print("\nGet your API key from: https://openrouter.ai/")
        return 1
    
    # Parse bias levels if provided
    if args.levels:
        try:
            levels = [int(x.strip()) for x in args.levels.split(',')]
        except ValueError:
            print("ERROR: Invalid levels format. Use comma-separated integers (e.g., '0,25,50,75,100')")
            return 1
    else:
        levels = None  # Use default range 0-100 with step 5
    
    # Determine which biases to test
    if args.bias == 'all':
        bias_types = ['authority', 'bandwagon', 'framing', 'confirmation']
    else:
        bias_types = [args.bias]
    
    print(f"\n{'='*60}")
    print(f"STARTING API-BASED BIAS EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Bias types: {bias_types}")
    print(f"Samples per prompt: {args.samples}")
    print(f"Permutations: {args.permutations}")
    print(f"Reasoning temperature: {args.reasoning_temp}")
    print(f"Choice temperature: {args.choice_temp}")
    if levels:
        print(f"Custom levels: {levels}")
    print(f"{'='*60}")
    
    # Run experiments for each bias type
    total_start_time = time.time()
    
    for bias_type in bias_types:
        print(f"\n\n🔬 TESTING {bias_type.upper()} BIAS")
        print("="*50)
        
        bias_start_time = time.time()
        
        try:
            # Load scenarios and options
            scenarios, mcq_options, prompt_template = load_scenarios_and_options(bias_type)
            
            if not scenarios:
                print(f"❌ No scenarios found for {bias_type} bias")
                continue
            
            print(f"📋 Loaded {len(scenarios)} scenarios for {bias_type} bias")
            
            # Create plot directory
            plot_dir = f"./api_{args.model}_{bias_type}_plots"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Initialize experiment
            experiment = APIPromptControlExperiment(
                api_key=api_key,
                model_name=args.model,
                plot_dir=plot_dir,
                bias_type=bias_type,
                likert_levels=levels,
                is_testing_mode=args.test_mode,
                num_permutations=args.permutations,
                reasoning_temperature=args.reasoning_temp,
                choice_temperature=args.choice_temp,
                num_samples_per_prompt=args.samples,
                max_workers=args.max_workers
            )
            
            # Run experiment
            experiment.run(scenarios, mcq_options, prompt_template, bias_type)
            
            bias_duration = time.time() - bias_start_time
            print(f"✅ {bias_type.title()} bias experiment completed in {bias_duration:.1f}s")
            
        except Exception as e:
            print(f"❌ Error running {bias_type} experiment: {e}")
            if args.test_mode:
                import traceback
                traceback.print_exc()
            continue
    
    total_duration = time.time() - total_start_time
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED in {total_duration:.1f}s")
    print(f"📊 Results saved to respective plot directories")

if __name__ == "__main__":
    sys.exit(main())
