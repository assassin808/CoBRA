# bias_with_shared_training.py - Optimized bias experiments with shared training pipeline

import argparse
import os
import sys
import json
import time
import torch

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

class SharedTrainingBiasExperiments:
    """Class to manage bias experiments with shared training pipeline."""
    
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
        self.rep_reading_pipeline = None
        self.rep_reader = None
        self.control_layers = None
        self.dataset = None
        self.all_experiments = None
        
    def train_shared_pipeline(self):
        """Train the RepE pipeline and RepReader once for reuse across experiments."""
        print(f"\n{'='*60}")
        print(f"TRAINING SHARED PIPELINE FOR {self.bias_type.upper()} BIAS")
        print(f"{'='*60}")
        
        # --- 1. Model Loading ---
        print("--- 1. Loading Model and Tokenizer ---")
        if self.model_name:
            model_name_or_path = self.data_manager.get_model_path(self.bias_type, self.model_name)
            print(f"Using specified model: {self.model_name} -> {model_name_or_path}")
        else:
            model_name_or_path = self.data_manager.get_model_path(self.bias_type)
            print(f"Using default model: {model_name_or_path}")
        
        self.model, self.tokenizer, self.user_tag, self.assistant_tag, self.assistant_prompt_for_choice, self.device = load_model_and_tags(model_name_or_path)
        if self.model is None:
            print("Model loading failed. Exiting.")
            return False
        if self.is_testing_mode:
            print(f"[DEBUG] Model loaded on device: {self.device}")

        # --- 2. RepReader Setup ---
        print("\n--- 2. Setting up RepReader Pipeline ---")
        rep_token = -1
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers - 1, -1))
        self.rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)

        # --- 3. Dataset Preparation ---
        print(f"\n--- 3. Preparing Generated {self.bias_type.title()} Dataset ---")
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
        except Exception as e:
            print(f"Failed to load training dataset for {self.bias_type}: {e}")
            return False

        # --- 4. RepReader Training ---
        print(f"\n--- 4. Training {self.bias_type.title()} Bias RepReader ---")
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
        print("RepReader trained.")

        # --- 5. RepReader Accuracy Evaluation ---
        print("\n--- 5. Evaluating RepReader Accuracy ---")
        base_plot_dir = ensure_plot_dir(f"{self.bias_type}_plots")
        results, best_layer = evaluate_repread_accuracy(
            self.rep_reading_pipeline,
            self.dataset,
            rep_token,
            hidden_layers,
            self.rep_reader,
            base_plot_dir,
            debug=self.is_testing_mode
        )
        
        # Store control layers for experiments - ensure they stay within available layer range
        # best_layer should be negative (e.g., -15), hidden_layers are [-1, -2, ..., -32]
        max_layer = max(hidden_layers)  # -1
        min_layer = min(hidden_layers)  # -32
        
        # Calculate control layer range around best_layer, but constrain to available layers
        control_start = max(best_layer - 3, min_layer)
        control_end = min(best_layer + 4, max_layer + 1)  # +1 for range end
        self.control_layers = list(range(control_start, control_end))
        
        # Ensure control_layers only contains layers that exist in RepReader directions
        self.control_layers = [layer for layer in self.control_layers if layer in self.rep_reader.directions]
        
        print(f"Best layer: {best_layer}")
        print(f"Available layers: {min_layer} to {max_layer}")
        print(f"Using control layers: {self.control_layers}")

        # --- 6. Load Original Experiment Data ---
        print(f"\n--- 6. Loading Original {self.bias_type.title()} Scenarios for Testing ---")
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
    
    def run_experiments_with_temperature(self, temperature, num_permutations=3):
        """Run experiments with a specific temperature using the pre-trained pipeline."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENTS WITH TEMPERATURE {temperature}")
        print(f"{'='*60}")
        
        if self.rep_reader is None:
            print("Error: Shared pipeline not trained yet. Call train_shared_pipeline() first.")
            return False
        
        # Create temperature-specific suffix and directories
        temp_suffix = f"_temp{temperature}"
        plot_dir_name = f"{self.bias_type}_plots{temp_suffix}"
        PLOT_DIR = ensure_plot_dir(plot_dir_name)
        
        # --- RepE Control Experiments ---
        print(f"\n--- 7. RepE Control Experiments (Temperature {temperature}) ---")
        rep_control_pipeline = pipeline(
            "rep-control", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            layers=self.control_layers, 
            control_method="reading_vec"
        )
        
        repe_experiment = RepEControlExperiment(
            self.model, self.tokenizer, self.device, PLOT_DIR, 
            rep_control_pipeline, self.rep_reader, self.control_layers,
            is_testing_mode=self.is_testing_mode, num_permutations=num_permutations,
            experiment_suffix=temp_suffix,
            reasoning_temperature=temperature,  # Pass temperature
            choice_temperature=temperature      # Pass temperature
        )
        
        # Run experiments for each original dataset
        for exp_name, exp_data in self.all_experiments.items():
            print(f"\n--- Running RepE Control for {self.bias_type.title()} - {exp_name.title()} (T={temperature}) ---")
            try:
                repe_experiment.run(
                    exp_data['scenarios'], 
                    exp_data['mcq_options'], 
                    exp_data['prompt_template'], 
                    f"{self.bias_type.title()} ({exp_name.title()})"
                )
            except Exception as e:
                print(f"Error in RepE experiment for {exp_name}: {e}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error args: {e.args}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                continue

        # --- Prompt-Based Control Experiments ---
        print(f"\n--- 8. Prompt-Based Control Experiments (Temperature {temperature}) ---")
        
        # Define percentage scale levels
        likert_levels = list(range(0, 101, 5))  # [0, 5, 10, 15, ..., 100]
        
        prompt_experiment = PromptControlExperiment(
            self.model, self.tokenizer, self.device, PLOT_DIR,
            bias_type=self.bias_type,
            likert_levels=likert_levels,
            is_testing_mode=self.is_testing_mode, 
            num_permutations=num_permutations,
            reasoning_temperature=temperature,
            choice_temperature=temperature,
            experiment_suffix=temp_suffix
        )
        
        # Run prompt experiments for each original dataset
        for exp_name, exp_data in self.all_experiments.items():
            print(f"\n--- Running Prompt Control for {self.bias_type.title()} - {exp_name.title()} (T={temperature}) ---")
            try:
                prompt_experiment.run(
                    exp_data['scenarios'], 
                    exp_data['mcq_options'], 
                    exp_data['prompt_template'], 
                    f"{self.bias_type.title()} ({exp_name.title()})"
                )
            except Exception as e:
                print(f"Error in Prompt experiment for {exp_name}: {e}")
                continue
        
        print(f"✓ All experiments completed for temperature {temperature}")
        return True
    
    def cleanup(self):
        """Clean up GPU memory after experiments."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'rep_reading_pipeline') and self.rep_reading_pipeline is not None:
            del self.rep_reading_pipeline
        torch.cuda.empty_cache()
        print("✓ GPU memory cleaned up")


def run_temperature_ablation_optimized(bias_types=None, temperatures=None, model_name="mistral-7b-local", is_testing_mode=False, num_permutations=1):
    """
    Optimized temperature ablation that trains once per bias type and reuses for all temperatures.
    """
    
    # Default values
    if bias_types is None:
        bias_types = ['authority']  # Default to authority only
    
    if temperatures is None:
        temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Create results directory
    results_dir = f"temperature_ablation_results_optimized"
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("OPTIMIZED TEMPERATURE ABLATION STUDY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Bias types: {bias_types}")
    print(f"Temperature values: {temperatures}")
    print(f"Optimization: Train once per bias type, reuse for all temperatures")
    print("="*60)
    
    # Process each bias type
    for bias_type in bias_types:
        print(f"\n{'='*40}")
        print(f"PROCESSING: {bias_type.upper()}")
        print(f"{'='*40}")
        
        # Initialize shared training pipeline
        shared_experiments = SharedTrainingBiasExperiments(
            bias_type=bias_type,
            model_name=model_name,
            is_testing_mode=is_testing_mode
        )
        
        # Train shared pipeline once
        print(f"Training shared pipeline for {bias_type}...")
        training_start = time.time()
        
        try:
            success = shared_experiments.train_shared_pipeline()
            training_duration = time.time() - training_start
            
            if not success:
                print(f"✗ Failed to train shared pipeline for {bias_type}")
                continue
                
            print(f"✓ Pipeline trained in {training_duration:.2f}s")
            
        except Exception as e:
            print(f"✗ Training failed for {bias_type}: {e}")
            continue
        
        # Run experiments for each temperature
        for temperature in temperatures:
            print(f"\n--- Running {bias_type} with T={temperature} ---")
            
            try:
                success = shared_experiments.run_experiments_with_temperature(
                    temperature=temperature,
                    num_permutations=num_permutations
                )
                
                status = "✓" if success else "✗"
                print(f"{status} {bias_type} T={temperature} completed")
                
            except Exception as e:
                print(f"✗ {bias_type} T={temperature} failed: {e}")
        
        # Cleanup GPU memory before next bias type
        shared_experiments.cleanup()
    
    print(f"\n✓ All experiments completed!")
    print(f"Results saved in: {results_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run optimized temperature ablation study')
    parser.add_argument('--bias', type=str, default='authority',
                       help='Bias type(s) to test (default: authority)')
    parser.add_argument('--temperatures', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                       help='Comma-separated list of temperature values')
    parser.add_argument('--model', type=str, default='mistral-7b-local',
                       help='Model to use (default: mistral-7b-local)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (fewer scenarios)')
    parser.add_argument('--permutations', type=int, default=1,
                       help='Number of permutations per experiment (default: 1)')
    
    args = parser.parse_args()
    
    # Parse bias types
    if args.bias == 'all':
        bias_types = ['authority', 'bandwagon', 'framing', 'confirmation']
    else:
        bias_types = [b.strip() for b in args.bias.split(',')]
    
    # Parse temperatures
    temperatures = [float(t.strip()) for t in args.temperatures.split(',')]
    
    # Run the optimized ablation study
    run_temperature_ablation_optimized(
        bias_types=bias_types,
        temperatures=temperatures,
        model_name=args.model,
        is_testing_mode=args.test,
        num_permutations=args.permutations
    )
    
    print("Optimized ablation study completed!")


if __name__ == "__main__":
    main()
