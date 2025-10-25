#!/usr/bin/env python3
"""
run_persona_ablation.py - Run persona ablation experiments for authority bias

This script runs bias experiments across different personas to understand how 
personality characteristics affect susceptibility to authority bias.
"""

import argparse
import os
import sys
import json
import time
import torch
import traceback

# Add control modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import pipeline
from control.repe import repe_pipeline_registry
repe_pipeline_registry()

from control.repe_experiment import RepEControlExperiment
from control.prompt_experiment import PromptControlExperiment
from control.experiment_utils import load_model_and_tags, evaluate_repread_accuracy
from control.env_config import BATCH_SIZE, NUM_PERMUTATIONS

from utils_bias import BiasDataManager, ensure_plot_dir

class PersonabiasExperiments:
    """Class to manage bias experiments across different personas."""
    
    def __init__(self, bias_type, model_name=None, is_testing_mode=False):
        self.bias_type = bias_type
        self.model_name = model_name
        self.is_testing_mode = is_testing_mode
        
        # Initialize data manager
        self.data_manager = BiasDataManager(os.path.dirname(__file__))
        
        # Training pipeline components (populated during training)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.user_tag = None
        self.assistant_tag = None
        self.assistant_prompt_for_choice = None
        self.rep_reading_pipeline = None
        self.rep_reader = None
        self.control_layers = None
        self.dataset = None
        self.all_experiments = None
        
    def load_personas(self, persona_file):
        """Load personas from JSON file."""
        with open(persona_file, 'r') as f:
            return json.load(f)
    
    def apply_persona_to_scenario(self, scenario, persona_description, persona_name):
        """Apply persona prefix to a scenario dictionary with improved format."""
        if isinstance(scenario, dict):
            # Create a copy of the scenario and modify the statement or add a persona field
            modified_scenario = scenario.copy()
            
            # For authority scenarios, we can add persona context to the statement
            if 'statement' in scenario:
                persona_prefix = f"Pretend you are {persona_name}. {persona_description}. "
                modified_scenario['statement'] = persona_prefix + scenario['statement']
            
            return modified_scenario
        else:
            # If it's a string, treat it as before
            persona_prefix = f"Pretend you are {persona_name}. {persona_description}. "
            return persona_prefix + scenario
    
    def train_shared_pipeline(self):
        """Train the RepE pipeline and RepReader once for reuse across personas."""
        print(f"\n{'='*60}")
        print(f"TRAINING SHARED PIPELINE FOR {self.bias_type.upper()} BIAS")
        print(f"{'='*60}")
        
        # Load model and tokenizer
        print("\n--- 1. Loading Model ---")
        try:
            if self.model_name:
                # If an absolute path was provided, use it directly
                if os.path.isabs(self.model_name):
                    model_name_or_path = self.model_name
                    print(f"Using absolute model path: {model_name_or_path}")
                else:
                    model_name_or_path = self.data_manager.get_model_path(self.bias_type, self.model_name)
                    print(f"Using specified model: {self.model_name} -> {model_name_or_path}")
            else:
                model_name_or_path = self.data_manager.get_model_path(self.bias_type)
                print(f"Using default model: {model_name_or_path}")
            
            self.model, self.tokenizer, self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice, self.device = load_model_and_tags(model_name_or_path)
            if self.model is None:
                print("Model loading failed. Exiting.")
                return False
            print(f"✓ Model loaded: {self.model_name}")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False
        
        # Initialize control layer candidates (used for RepReader training).
        # Final control_layers will be selected after evaluating RepReader.
        print("\n--- 2. Setting up Control Layers ---")
        if self.model is None:
            print("✗ Model is None, cannot access config")
            return False
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        print(f"✓ Candidate hidden layers: {hidden_layers}")
        
        # Create RepE reading pipeline
        print("\n--- 3. Creating RepE Reading Pipeline ---")
        try:
            self.rep_reading_pipeline = pipeline(
                "rep-reading",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            print("✓ RepE reading pipeline created")
        except Exception as e:
            print(f"✗ RepE pipeline creation failed: {e}")
            return False
        
        # Load experimental datasets
        print("\n--- 4. Loading Experimental Datasets ---")
        try:
            self.dataset = self.data_manager.create_training_dataset(
                self.bias_type, self.tokenizer, user_tag=self.user_tag, assistant_tag=self.assistant_tag, 
                testing=self.is_testing_mode, model_name=self.model_name
            )
            if self.is_testing_mode:
                print(f"[DEBUG] Training data size: {len(self.dataset['train']['data'])}. Test data size: {len(self.dataset['test']['data'])}.")
            if not self.dataset['train']['data']:
                print("Dataset preparation failed. Exiting.")
                return False
            print(f"✓ Loaded {len(self.dataset['train']['data'])} training examples for {self.bias_type}")
        except Exception as e:
            print(f"✗ Dataset loading failed: {e}")
            return False
        
        # Train RepReader
        print("\n--- 5. Training RepReader ---")
        try:
            # Define rep_token and use candidate hidden_layers defined above
            rep_token = -1
            # Use the candidate hidden_layers variable created earlier (not self.control_layers)
            # hidden_layers is already set to the candidate list above
            self.rep_reader = self.rep_reading_pipeline.get_directions(
                self.dataset['train']['data'],
                rep_token=rep_token,
                hidden_layers=hidden_layers,
                train_labels=self.dataset['train']['labels'],
                direction_method='pca',
                n_difference=1,
                batch_size=BATCH_SIZE,
                direction_finder_kwargs={'n_components': 4},
            )
            print("✓ RepReader trained successfully")

            # Evaluate RepReader accuracy and select control layers (top-K by score).
            print("\n--- 6. Evaluating RepReader Accuracy and selecting control layers ---")
            plot_dir = ensure_plot_dir(f"{self.bias_type}_plots")
            try:
                results, best_layer = evaluate_repread_accuracy(
                    self.rep_reading_pipeline,
                    self.dataset,
                    rep_token,
                    hidden_layers,
                    self.rep_reader,
                    plot_dir,
                    debug=self.is_testing_mode
                )
                # If we have valid results and the best_layer exists in directions, pick top-K
                if results and best_layer in self.rep_reader.directions:
                    top_k = 15
                    self.control_layers = sorted(self.rep_reader.directions.keys(), key=lambda l: results.get(l, 0), reverse=True)[:top_k]
                    print(f"✓ Selected control layers (top {top_k}): {self.control_layers}")
                else:
                    # Fallback: use candidate hidden layers filtered by available directions
                    self.control_layers = [l for l in hidden_layers if l in self.rep_reader.directions]
                    print(f"⚠️  Using fallback control layers: {self.control_layers}")
            except Exception as e:
                print(f"✗ RepReader evaluation failed: {e}")
                self.control_layers = [l for l in hidden_layers if l in self.rep_reader.directions]
                print(f"⚠️  Fallback control layers: {self.control_layers}")
        except Exception as e:
            print(f"✗ RepReader training failed: {e}")
            traceback.print_exc()
            return False
        
        # Load experimental scenarios
        print("\n--- 6. Loading Experiment Scenarios ---")
        num_scenarios = 4 if self.is_testing_mode else None
        experiment_names = self.data_manager.get_experiment_names(self.bias_type)
        
        self.all_experiments = {}
        for exp_name in experiment_names:
            try:
                scenarios = self.data_manager.load_experiment_scenarios(
                    self.bias_type, exp_name, num_scenarios=num_scenarios
                )
                mcq_options, prompt_template = self.data_manager.get_mcq_options_and_templates(self.bias_type, exp_name)
                
                self.all_experiments[exp_name] = {
                    'scenarios': scenarios,
                    'mcq_options': mcq_options,
                    'prompt_template': prompt_template
                }
                print(f"✓ Loaded {len(scenarios)} scenarios for {exp_name}")
            except Exception as e:
                print(f"✗ Failed to load {exp_name}: {e}")
                continue
        
        print(f"\n✓ Shared training pipeline setup complete!")
        print(f"✓ RepReader trained and ready for reuse")
        print(f"✓ {len(self.all_experiments)} experiment datasets loaded")
        return True

    def prepare_baseline(self):
        """Prepare minimal state for baseline-only runs: load model/tokenizer and scenarios, skip RepReader/RepE."""
        print(f"\nPreparing baseline-only runtime for {self.bias_type} (no RepReader training)")
        try:
            if self.model_name:
                if os.path.isabs(self.model_name):
                    model_name_or_path = self.model_name
                    print(f"Using absolute model path: {model_name_or_path}")
                else:
                    model_name_or_path = self.data_manager.get_model_path(self.bias_type, self.model_name)
                    print(f"Using specified model: {self.model_name} -> {model_name_or_path}")
            else:
                model_name_or_path = self.data_manager.get_model_path(self.bias_type)
                print(f"Using default model: {model_name_or_path}")

            self.model, self.tokenizer, self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice, self.device = load_model_and_tags(model_name_or_path)
            if self.model is None:
                print("Model loading failed for baseline-only run.")
                return False
            print("✓ Model/tokenizer loaded for baseline-only run")
        except Exception as e:
            print(f"✗ Error loading model/tokenizer for baseline-only: {e}")
            return False

        # Load experimental scenarios (no RepReader needed)
        num_scenarios = 4 if self.is_testing_mode else None
        experiment_names = self.data_manager.get_experiment_names(self.bias_type)
        self.all_experiments = {}
        for exp_name in experiment_names:
            try:
                scenarios = self.data_manager.load_experiment_scenarios(
                    self.bias_type, exp_name, num_scenarios, None
                )
                mcq_options, prompt_template = self.data_manager.get_mcq_options_and_templates(self.bias_type, exp_name)
                self.all_experiments[exp_name] = {
                    'scenarios': scenarios,
                    'mcq_options': mcq_options,
                    'prompt_template': prompt_template
                }
                print(f"✓ Loaded {len(scenarios)} scenarios for {exp_name} (baseline-only)")
            except Exception as e:
                print(f"✗ Failed to load {exp_name} (baseline-only): {e}")
                continue

        return True
    
    def run_experiments_with_persona(self, persona_name, persona_description, num_permutations=1, baseline_only=False):
        """Run experiments with a specific persona using the pre-trained pipeline."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENTS WITH PERSONA: {persona_name}")
        print(f"{'='*60}")
        
        if self.rep_reader is None and not baseline_only:
            print("Error: Shared pipeline not trained yet. Call train_shared_pipeline() first.")
            return False
        
        # Create persona-specific suffix and directories  
        safe_persona_name = persona_name.replace(' ', '_').replace('.', '')
        persona_suffix = f"_persona_{safe_persona_name}"
        plot_dir_name = f"{self.bias_type}_plots{persona_suffix}"
        PLOT_DIR = ensure_plot_dir(plot_dir_name)
        
        # --- RepE Control Experiments ---
        # If baseline_only is requested, skip RepE entirely
        if baseline_only:
            print("Skipping RepE control experiments due to --baseline-only flag.")
        else:
            print(f"\n--- RepE Control Experiments (Persona: {persona_name}) ---")

            # Create single rep_control_pipeline with reading_vec method
            rep_control_pipeline = pipeline(
                "rep-control",
                model=self.model,
                tokenizer=self.tokenizer,
                layers=self.control_layers,
                control_method="reading_vec"
            )

            # Create RepE experiment instance
            repe_experiment = RepEControlExperiment(
                self.model,
                self.tokenizer, 
                self.device,
                PLOT_DIR,
                rep_control_pipeline,
                self.rep_reader,
                self.control_layers,
                is_testing_mode=self.is_testing_mode,
                num_permutations=num_permutations,
                experiment_suffix=persona_suffix
            )

            # Run experiments for each dataset with all RepE methods
            for exp_name, exp_data in self.all_experiments.items():
                print(f"\n--- Running RepE Control for {exp_name} with {persona_name} ---")

                # Apply persona to all scenarios
                modified_scenarios = []
                for scenario in exp_data['scenarios']:
                    modified_scenario = self.apply_persona_to_scenario(scenario, persona_description, persona_name)
                    modified_scenarios.append(modified_scenario)

                try:
                    repe_experiment.run(
                        modified_scenarios,
                        exp_data['mcq_options'],
                        exp_data['prompt_template'],
                        f"{exp_name.title()} (Persona: {persona_name})",
                        operators=['linear_comb', 'projection', 'orthognalize']
                    )
                    print(f"✓ Completed RepE control for {exp_name}")

                except Exception as e:
                    print(f"✗ Failed RepE control for {exp_name}: {e}")
                    continue
        
    # --- Prompt Control Experiments ---
        print(f"\n--- Prompt Control Experiments (Persona: {persona_name}) ---")
        
        for exp_name, exp_data in self.all_experiments.items():
            print(f"\nExperiment: {exp_name}")
            
            # Apply persona to all scenarios
            modified_scenarios = []
            for scenario in exp_data['scenarios']:
                modified_scenario = self.apply_persona_to_scenario(scenario, persona_description, persona_name)
                modified_scenarios.append(modified_scenario)
            
            try:
                prompt_experiment = PromptControlExperiment(
                    self.model,
                    self.tokenizer,
                    self.device,
                    PLOT_DIR,
                    bias_type=self.bias_type,
                    is_testing_mode=self.is_testing_mode,
                    num_permutations=num_permutations,
                    experiment_suffix=persona_suffix
                )

                # For baseline-only runs, scenarios were prepared by prepare_baseline and
                # available in self.all_experiments. Run natural-language baseline for
                # framing and bandwagon only, then optionally run the standard prompt
                # experiment if baseline_only is False.
                if self.bias_type in ('framing', 'bandwagon'):
                    orig_gen = getattr(prompt_experiment, '_generate_likert_control_prompt', None)
                    try:
                        prompt_experiment._generate_likert_control_prompt = lambda level: f"Pretend you have complete {self.bias_type} bias."
                        baseline_suffix = persona_suffix + '_baseline_natural'
                        prompt_experiment.experiment_suffix = baseline_suffix
                        print(f"Running persona natural-language baseline for {exp_name} (Persona: {persona_name}) with suffix: {baseline_suffix}")
                        prompt_experiment.run(
                            modified_scenarios,
                            exp_data['mcq_options'],
                            exp_data['prompt_template'],
                            f"{exp_name.title()} (Persona: {persona_name})"
                        )
                    finally:
                        if orig_gen is not None:
                            prompt_experiment._generate_likert_control_prompt = orig_gen
                        prompt_experiment.experiment_suffix = persona_suffix

                if not baseline_only:
                    prompt_experiment.run(
                        modified_scenarios,
                        exp_data['mcq_options'],
                        exp_data['prompt_template'],
                        f"{exp_name.title()} (Persona: {persona_name})"
                    )
                else:
                    print(f"Baseline-only: skipped standard prompt experiment for {exp_name} (Persona: {persona_name})")

                print(f"✓ Completed prompt control for {exp_name}")

            except Exception as e:
                print(f"✗ Failed prompt control for {exp_name}: {e}")
                continue
        
        print(f"\n✓ All experiments completed for persona: {persona_name}")
        return True


def run_persona_ablation(bias_types, personas, selected_persona_names, model_name='mistral-7b-local', is_testing_mode=False, num_permutations=1, baseline_only=False):
    """Run persona ablation study across multiple personas."""
    
    for bias_type in bias_types:
        print(f"\n{'='*80}")
        print(f"STARTING PERSONA ABLATION FOR {bias_type.upper()} BIAS")
        print(f"Model: {model_name}")
        print(f"Personas: {selected_persona_names}")
        print(f"Testing Mode: {is_testing_mode}")
        print(f"{'='*80}")
        
        # Initialize experiment manager
        experiment_manager = PersonabiasExperiments(
            bias_type=bias_type,
            model_name=model_name,
            is_testing_mode=is_testing_mode
        )
        
        # Train shared pipeline once (or prepare minimal baseline state)
        if baseline_only:
            print(f"\n🧠 Preparing baseline-only runtime for {bias_type}...")
            if not experiment_manager.prepare_baseline():
                print(f"Failed to prepare baseline runtime for {bias_type}, skipping...")
                continue
        else:
            print(f"\n🧠 Training shared pipeline for {bias_type}...")
            if not experiment_manager.train_shared_pipeline():
                print(f"Failed to train pipeline for {bias_type}, skipping...")
                continue
        
        # Run experiments for each selected persona
        total_personas = len(selected_persona_names)
        for i, persona_name in enumerate(selected_persona_names, 1):
            if persona_name not in personas:
                print(f"Warning: Persona '{persona_name}' not found in personas file, skipping...")
                continue

            print(f"\n🎭 [{i}/{total_personas}] Running experiments for {persona_name}...")

            start_time = time.time()
            success = experiment_manager.run_experiments_with_persona(
                persona_name=persona_name,
                persona_description=personas[persona_name],
                num_permutations=num_permutations,
                baseline_only=baseline_only
            )
            end_time = time.time()

            if success:
                print(f"✅ Completed {persona_name} in {end_time - start_time:.1f}s")
            else:
                print(f"❌ Failed {persona_name}")

        print(f"\n🎉 Persona ablation completed for {bias_type}!")


def main():
    parser = argparse.ArgumentParser(description='Run persona ablation study for bias experiments')
    parser.add_argument('--bias', type=str, default='authority',
                       help='Bias type(s) to test (default: authority)')
    parser.add_argument('--personas', type=str, 
                       default='Adam Smith,Giorgio Rossi,Ayesha Khan,Carlos Gomez,Francisco Lopez,Abigail Chen,Arthur Burton,Carmen Ortiz,Hailey Johnson,Isabella Rodriguez',
                       help='Comma-separated list of persona names (default: 10 diverse personas)')
    parser.add_argument('--persona-file', type=str, default='./personas_extracted.json',
                       help='Path to personas JSON file (default: ./personas_extracted.json)')
    parser.add_argument('--model', type=str, default='mistral-7b-local',
                       help='Model to use (default: mistral-7b-local)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (fewer scenarios)')
    parser.add_argument('--permutations', type=int, default=1,
                       help='Number of permutations per experiment (default: 1)')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run only per-person natural-language baselines and skip RepE and standard prompt experiments')
    
    args = parser.parse_args()
    
    # Parse bias types
    if args.bias == 'all':
        bias_types = ['authority', 'bandwagon', 'framing', 'confirmation']
    else:
        bias_types = [b.strip() for b in args.bias.split(',')]
    
    # Parse persona names
    selected_persona_names = [p.strip() for p in args.personas.split(',')]
    
    # Load personas
    print(f"Loading personas from: {args.persona_file}")
    try:
        with open(args.persona_file, 'r') as f:
            personas = json.load(f)
        print(f"✓ Loaded {len(personas)} personas")
    except Exception as e:
        print(f"✗ Failed to load personas: {e}")
        return
    
    # Validate selected personas
    invalid_personas = [p for p in selected_persona_names if p not in personas]
    if invalid_personas:
        print(f"Warning: These personas were not found: {invalid_personas}")
        available_personas = list(personas.keys())
        print(f"Available personas: {available_personas}")
        return
    
    print(f"\n📋 Experiment Configuration:")
    print(f"  Bias types: {bias_types}")
    print(f"  Personas: {selected_persona_names}")
    print(f"  Model: {args.model}")
    print(f"  Test mode: {args.test}")
    print(f"  Permutations: {args.permutations}")
    
    # Run the persona ablation study
    run_persona_ablation(
        bias_types=bias_types,
        personas=personas,
        selected_persona_names=selected_persona_names,
        model_name=args.model,
        is_testing_mode=args.test,
        num_permutations=args.permutations,
        baseline_only=args.baseline_only,
    )
    
    print(f"\n🎉 All persona ablation experiments completed!")
    print(f"Check the results directories for output files.")


if __name__ == "__main__":
    main()
