# utils_bias.py - Unified utilities for all bias experiments

import os
import sys
import json
from typing import Dict, List, Optional, Tuple

# Add bias-specific directories to Python path
current_dir = os.path.dirname(__file__)
examples_dir = os.path.dirname(current_dir)

authority_dir = os.path.join(examples_dir, 'authority')
bandwagon_dir = os.path.join(examples_dir, 'bandwagon')
framing_dir = os.path.join(examples_dir, 'framing')
confirmation_dir = os.path.join(examples_dir, 'confirmation')

# Add to sys.path if not already there
for path in [authority_dir, bandwagon_dir, framing_dir, confirmation_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Import bias-specific utilities
try:
    from utils_authority import (
        create_authority_dataset_from_generated, 
        load_authority_scenarios,
        load_milgram_scenarios,
        load_stanford_prison_scenarios
    )
except ImportError as e:
    print(f"Warning: Could not import authority utils: {e}")
    create_authority_dataset_from_generated = None
    load_authority_scenarios = None

try:
    from utils_bandwagon import (
        create_bandwagon_dataset_from_generated,
        load_bandwagon_scenarios
    )
except ImportError as e:
    print(f"Warning: Could not import bandwagon utils: {e}")
    create_bandwagon_dataset_from_generated = None
    load_bandwagon_scenarios = None

try:
    from utils_framing import (
        create_framing_dataset_from_generated,
        load_framing_scenarios
    )
except ImportError as e:
    print(f"Warning: Could not import framing utils: {e}")
    create_framing_dataset_from_generated = None
    load_framing_scenarios = None

try:
    from utils_confirmation import (
        create_confirmation_dataset_from_generated,
        load_confirmation_scenarios
    )
except ImportError as e:
    print(f"Warning: Could not import confirmation utils: {e}")
    create_confirmation_dataset_from_generated = None
    load_confirmation_scenarios = None

class BiasDataManager:
    """Manages data loading and dataset creation for all bias types"""
    
    def __init__(self, base_dir, config_file=None, model_config=None):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, '..', '..', 'data')
        self.generated_dir = os.path.join(base_dir, '..', '..', 'data_generated')
        
        # Load configuration from file or use provided config
        if config_file:
            self.config = self._load_config_file(config_file)
            self.model_config = self.config.get('models', {})
        elif model_config:
            self.model_config = model_config
            self.config = {'models': model_config}
        else:
            # Load default config file if it exists, otherwise use hardcoded defaults
            default_config_path = os.path.join(current_dir, 'model_config.json')
            if os.path.exists(default_config_path):
                self.config = self._load_config_file(default_config_path)
                self.model_config = self.config.get('models', {})
            else:
                self.model_config = self._get_default_model_config()
                self.config = {'models': self.model_config}
    
    def _load_config_file(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_file}: {e}")
    
    def _get_default_model_config(self):
        """Get default model configuration with both local and HuggingFace models"""
        return {
            # Local models (already downloaded)
            'mistral-7b-local': {
                'path': "../../../mistral-7B-Instruct-v0.3",
                'type': 'local',
                'description': 'Mistral 7B Instruct v0.3 (Local)',
                'final_test': False
            },
            'qwen3-8b-local': {
                'path': "../../../mistral-7B-Instruct-v0.3", 
                'type': 'local',
                'description': 'Qwen3 8B Model (Local)',
                'final_test': False
            },
            
            # Final test models - HuggingFace
            'llama-3.1-8b': {
                'path': "meta-llama/Llama-3.1-8B-Instruct",
                'type': 'huggingface', 
                'description': 'Llama 3.1 8B Instruct',
                'final_test': True,
                'gated': True,
                'access_url': 'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct'
            },
            'mistral-7b': {
                'path': "mistralai/Mistral-7B-Instruct-v0.3",
                'type': 'huggingface',
                'description': 'Mistral 7B Instruct v0.3',
                'final_test': True,
                'gated': False
            },
            'gpt-oss-20b': {
                'path': "openai/gpt-oss-20b",
                'type': 'huggingface',
                'description': 'GPT OSS 20B',
                'final_test': True,
                'gated': True,
                'access_url': 'https://huggingface.co/openai/gpt-oss-20b'
            },
            'qwen3-8b': {
                'path': "Qwen/Qwen3-8B",
                'type': 'huggingface',
                'description': 'Qwen3 30B A3B Instruct',
                'final_test': True,
                'gated': False  # May change
            },
            'deepseek-r1-8b': {
                'path': "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                'type': 'huggingface',
                'description': 'DeepSeek R1 Distill Qwen 8B',
                'final_test': True,
                'gated': False  # May change
            },
            
            # Other HuggingFace models (for development/testing)
            'qwen2.5-7b-hf': {
                'path': "Qwen/Qwen2.5-7B-Instruct",
                'type': 'huggingface',
                'description': 'Qwen 2.5 7B Instruct (HuggingFace)',
                'final_test': False
            },
            'gemma-2-9b-hf': {
                'path': "google/gemma-2-9b-it",
                'type': 'huggingface',
                'description': 'Gemma 2 9B Instruct (HuggingFace)',
                'final_test': False
            },
            'phi-3-mini-hf': {
                'path': "microsoft/Phi-3-mini-4k-instruct",
                'type': 'huggingface',
                'description': 'Phi-3 Mini 4K Instruct (HuggingFace)',
                'final_test': False
            },
            'llama-3.2-3b-hf': {
                'path': "meta-llama/Llama-3.2-3B-Instruct",
                'type': 'huggingface',
                'description': 'Llama 3.2 3B Instruct (HuggingFace)',
                'final_test': False
            }
        }

    def get_bias_config(self, bias_type, model_name=None):
        """Get configuration for a specific bias type with specified model"""
        # Use default model from config if none specified
        if model_name is None:
            model_name = self.config.get('default_model', 'mistral-7b-local')
        
        model_info = self.model_config.get(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.model_config.keys())}")
        
        # Check if model is enabled
        if not model_info.get('enabled', True):
            raise ValueError(f"Model {model_name} is disabled in configuration")
        
        configs = {
            'authority': {
                'generated_file': 'authority_generated*_with_responses.json',
                'original_experiments': {
                    'milgram': {
                        'file': 'authority/authority_MilgramS.json',
                        'type': 'milgram'
                    },
                    'stanford': {
                        'file': 'authority/authority_StanPri.json',
                        'type': 'stanford'
                    }
                },
                'create_dataset_func': create_authority_dataset_from_generated,
                'load_scenarios_func': load_authority_scenarios,
                'model_path': model_info['path'],
                'model_type': model_info['type'],
                'model_description': model_info['description']
            },
            'bandwagon': {
                'generated_file': 'bandwagon_generated*_with_responses.json',
                'original_experiments': {
                    'asch': {
                        'file': 'bandwagon/bandwagon_Asch.json',
                        'type': 'asch'
                    },
                    'hotel': {
                        'file': 'bandwagon/bandwagon_Hotel.json',
                        'type': 'hotel'
                    }
                },
                'create_dataset_func': create_bandwagon_dataset_from_generated,
                'load_scenarios_func': load_bandwagon_scenarios,
                'model_path': model_info['path'],
                'model_type': model_info['type'],
                'model_description': model_info['description']
            },
            'framing': {
                'generated_file': 'framing_generated*_with_responses.json',
                'original_experiments': {
                    'asian': {
                        'file': 'framing',  # Directory, not file
                        'type': 'asian_disease'
                    },
                    'invest': {
                        'file': 'framing',  # Directory, not file
                        'type': 'investment_decision'
                    }
                },
                'create_dataset_func': create_framing_dataset_from_generated,
                'load_scenarios_func': load_framing_scenarios,
                'model_path': model_info['path'],
                'model_type': model_info['type'],
                'model_description': model_info['description']
            },
            'confirmation': {
                'generated_file': 'confirmation_generated*_with_responses.json',
                'original_experiments': {
                    'bias_info': {
                        'file': 'confirmation/confirmation_BiasInfo.json',
                        'type': 'bias_info'
                    },
                    'wason': {
                        'file': 'confirmation/confirmation_Wason.json',
                        'type': 'wason'
                    }
                },
                'create_dataset_func': create_confirmation_dataset_from_generated,
                'load_scenarios_func': load_confirmation_scenarios,
                'model_path': model_info['path'],
                'model_type': model_info['type'],
                'model_description': model_info['description']
            }
        }
        return configs.get(bias_type)
    
    def _resolve_file_path(self, pattern):
        """Resolve file path, handling wildcards - returns list of all matching files"""
        if '*' in pattern:
            import glob
            pattern_path = os.path.join(self.generated_dir, pattern)
            matching_files = glob.glob(pattern_path)
            if not matching_files:
                raise FileNotFoundError(f"No files found matching pattern: {pattern_path}")
            # Return all matching files sorted by modification time (newest first)
            return sorted(matching_files, key=os.path.getmtime, reverse=True)
        else:
            return [os.path.join(self.generated_dir, pattern)]

    def create_training_dataset(self, bias_type, tokenizer, user_tag, assistant_tag, testing=False, model_name=None):
        """Create training dataset from generated data for RepReader training"""
        config = self.get_bias_config(bias_type, model_name)
        if not config:
            raise ValueError(f"Unknown bias type: {bias_type}")
        
        generated_paths = self._resolve_file_path(config['generated_file'])
        create_func = config['create_dataset_func']
        
        try:
            # If multiple files found, combine their datasets
            if len(generated_paths) == 1:
                dataset = create_func(
                    generated_paths[0], 
                    tokenizer, 
                    user_tag=user_tag, 
                    assistant_tag=assistant_tag, 
                    testing=testing
                )
                print(f"Using generated dataset from {generated_paths[0]}")
            else:
                # Combine datasets from multiple files
                combined_dataset = None
                print(f"Found {len(generated_paths)} matching files, combining datasets:")
                
                for path in generated_paths:
                    print(f"  - {path}")
                    dataset = create_func(
                        path, 
                        tokenizer, 
                        user_tag=user_tag, 
                        assistant_tag=assistant_tag, 
                        testing=testing
                    )
                    
                    if combined_dataset is None:
                        combined_dataset = dataset
                    else:
                        # Combine the training data
                        combined_dataset['train']['data'].extend(dataset['train']['data'])
                        combined_dataset['train']['labels'].extend(dataset['train']['labels'])
                        if 'val' in dataset:
                            if 'val' not in combined_dataset:
                                combined_dataset['val'] = dataset['val']
                            else:
                                combined_dataset['val']['data'].extend(dataset['val']['data'])
                                if 'labels' in dataset['val']:
                                    combined_dataset['val']['labels'].extend(dataset['val']['labels'])
                
                dataset = combined_dataset
                print(f"Combined dataset contains {len(dataset['train']['data'])} training samples and {len(dataset['train']['labels'])} labels")
            
            if not dataset['train']['data']:
                raise FileNotFoundError("Generated dataset is empty")
            
            # Verify dataset integrity
            n_data = len(dataset['train']['data'])
            n_labels = len(dataset['train']['labels'])
            print(f"Dataset integrity check: {n_data} data samples, {n_labels} label pairs")
            if n_data != 2 * n_labels:
                raise ValueError(f"Data/label mismatch: {n_data} data samples should equal 2 × {n_labels} labels = {2 * n_labels}")
            
            print(f"Model: {config['model_description']} ({config['model_path']})")
            return dataset
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Could not load generated dataset for {bias_type} ({e})")
            raise e
    
    def load_experiment_scenarios(self, bias_type, experiment_name, num_scenarios=None, model_name=None):
        """Load scenarios for a specific experiment within a bias type"""
        config = self.get_bias_config(bias_type, model_name)
        if not config:
            raise ValueError(f"Unknown bias type: {bias_type}")
        
        if experiment_name not in config['original_experiments']:
            raise ValueError(f"Unknown experiment '{experiment_name}' for bias type '{bias_type}'")
        
        exp_config = config['original_experiments'][experiment_name]
        data_path = os.path.join(self.data_dir, exp_config['file'])
        load_func = config['load_scenarios_func']
        
        # Handle different loading function signatures
        if bias_type == 'authority':
            return load_func(data_path, exp_config['type'], num_scenarios)
        elif bias_type == 'bandwagon':
            return load_func(data_path, scenario_type=exp_config['type'])
        elif bias_type == 'framing':
            scenarios = load_func(data_path, num_scenarios)
            # Filter by type for framing scenarios
            return [s for s in scenarios if s.get('type') == exp_config['type']]
        elif bias_type == 'confirmation':
            return load_func(data_path, num_scenarios)
    
    def get_model_path(self, bias_type, model_name=None):
        """Get the model path for a specific bias type"""
        config = self.get_bias_config(bias_type, model_name)
        return config['model_path'] if config else None
    
    def get_model_info(self, model_name):
        """Get detailed information about a specific model"""
        return self.model_config.get(model_name)
    
    def list_available_models(self):
        """List all available models with their types and descriptions"""
        models = {}
        for name, info in self.model_config.items():
            models[name] = {
                'type': info['type'],
                'description': info['description'],
                'path': info['path']
            }
        return models
    
    def get_enabled_models(self):
        """Get list of enabled models only"""
        return {name: info for name, info in self.model_config.items() if info.get('enabled', True)}
    
    def get_local_models(self):
        """Get list of local models only"""
        return {name: info for name, info in self.model_config.items() 
                if info['type'] == 'local' and info.get('enabled', True)}
    
    def get_huggingface_models(self):
        """Get list of HuggingFace models only"""
        return {name: info for name, info in self.model_config.items() 
                if info['type'] == 'huggingface' and info.get('enabled', True)}
    
    def get_final_test_models(self):
        """Get list of final test models only"""
        return {name: info for name, info in self.model_config.items() 
                if info.get('final_test', False) and info.get('enabled', True)}
    
    def get_all_bias_types(self):
        """Get all available bias types"""
        return ['authority', 'bandwagon', 'framing', 'confirmation']
    
    def should_run_all_models(self):
        """Check if configuration specifies to run all models"""
        return self.config.get('testing', {}).get('run_all_models', False)
    
    def get_max_scenarios(self):
        """Get maximum scenarios from configuration"""
        return self.config.get('testing', {}).get('max_scenarios', None)
    
    def get_output_dir(self):
        """Get output directory from configuration"""
        return self.config.get('testing', {}).get('output_dir', '../../results')
    
    def update_model_status(self, model_name, enabled=True):
        """Enable or disable a model in the configuration"""
        if model_name in self.model_config:
            self.model_config[model_name]['enabled'] = enabled
        else:
            raise ValueError(f"Model {model_name} not found in configuration")
    
    def save_config(self, config_file):
        """Save current configuration to file"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_experiment_names(self, bias_type, model_name=None):
        """Get list of experiment names for a bias type"""
        config = self.get_bias_config(bias_type, model_name)
        return list(config['original_experiments'].keys()) if config else []

    def get_mcq_options_and_templates(self, bias_type, experiment_name):
        """Get MCQ options and prompt templates for specific bias type and experiment"""
        # Import bias-specific modules to get their MCQ options
        if bias_type == 'authority':
            if experiment_name == 'milgram':
                mcq_options = {
                    "A": "I completely agree.",
                    "B": "I mostly agree but would like more evidence.",
                    "C": "I am neutral and need much more information.",
                    "D": "I mostly disagree; the claim seems unconvincing.",
                    "E": "I completely disagree; the claim is unpersuasive."
                }
                prompt_template = "According to {authority_figure}, {authority_context}: {statement}. The claim is that {rationale}. Which of the following best reflects your view?"
            elif experiment_name == 'stanford':
                mcq_options = {
                    "A": "Enforce strict compliance because I am the authority, and they must obey.",
                    "B": "Assert authority by requiring compliance, while briefly addressing their concerns.",
                    "C": "Balance authority and collaboration by working on a joint solution.",
                    "D": "Minimize authority by supporting the group's efforts and stepping back.",
                    "E": "Withdraw from the situation, because I believe I am unable to command them."
                }
                prompt_template = "You are assigned the {authority_role} in a simulated environment. You have authority over {subordinate_group}. Below is the situation you face: {situation}. Which of the following best reflects your response?"
        
        elif bias_type == 'bandwagon':
            if experiment_name == 'asch':
                mcq_options = {
                    "A": "I completely agree.",
                    "B": "I mostly agree with other participants that the claim is true, but would like more evidence.",
                    "C": "I'm neutral and need much more information.",
                    "D": "I mostly disagree with other participants; the claim seems unconvincing.",
                    "E": "I completely disagree with other participants; the claim is unpersuasive."
                }
                prompt_template = "You are in a room with several other participants who have a similar educational background. The claim is: {statement}. You notice that all of the other participants agree with this claim. Which of the following best reflects your view?"
            elif experiment_name == 'hotel':
                mcq_options = {
                    "A": "I will follow the majority and do the same.",
                    "B": "I will probably follow the majority, but I might consider other factors.",
                    "C": "I am undecided and need more information.",
                    "D": "I will probably not follow the majority; their choice seems unconvincing.",
                    "E": "I will not follow the majority and will do something different."
                }
                prompt_template = "{scenario} {majority_behavior} What would you most likely do?"
        
        elif bias_type == 'framing':
            if experiment_name == 'asian':
                mcq_options = {
                    "A": "I strongly prefer the positively framed option (Program A).",
                    "B": "I somewhat prefer the positively framed option (Program A).",
                    "C": "I slightly prefer the positively framed option (Program A).",
                    "D": "I am somewhat undecided between the two options.",
                    "E": "I see both options as equivalent and have no preference."
                }
                prompt_template = "You are presented with a scenario where {prompt}. Two programs are proposed to address the situation:\nProgram A:{options[A]}\nProgram B:{options[B]}\n\nWhich of the following best reflects your choice?"
            elif experiment_name == 'invest':
                mcq_options = {
                    "A": "I strongly prefer the positively framed option (Option A).",
                    "B": "I somewhat prefer the positively framed option (Option A).",
                    "C": "I slightly prefer the positively framed option (Option A).",
                    "D": "I am somewhat undecided between the two options.",
                    "E": "I see both options as equivalent and have no preference."
                }
                prompt_template = "You are considering {prompt}. The following statements describe the same situation with different wording:\nOption A: {options[A]}\nOption B: {options[B]}\n\nWhich option sounds more appealing or persuasive to you?"
        
        elif bias_type == 'confirmation':
            if experiment_name == 'bias_info':
                # For bias_info: measuring agreement with controversial topics
                mcq_options = {
                    "A": "Only choose articles that support my belief, ignoring opposing or neutral viewpoints.",
                    "B": "Mostly choose articles that support my belief but briefly consider opposing views.",
                    "C": "Equally consider articles that both support and challenge my belief.",
                    "D": "Prioritize articles that challenge my belief but still consider supporting views.",
                    "E": "Only choose articles that challenge my belief, ignoring supporting or neutral viewpoints."
                }
                prompt_template = "You have long believed that {topic}. Here are three pieces of information: A) {options[A]} B) {options[B]} C) {options[C]}. Which of the following best reflects your view?"
            elif experiment_name == 'wason':
                # For Wason selection: measuring logical reasoning approach
                mcq_options = {
                    "A": "Select options that can confirm the rule, while ignoring other possibilities.",
                    "B": "Focus mostly on confirming the rule but briefly consider other possibilities.",
                    "C": "Consider both confirming the rule and testing for cases where the rule might not apply.",
                    "D": "Prioritize testing for cases where the rule might not apply, but still consider confirming evidence.",
                    "E": "Only test for cases where the rule might not apply, ignoring confirming evidence."
                }
                prompt_template = "You are presented with a rule: '{rule}' Below are four options, and your task is to choose which options to test to determine if the rule is true. The options are as follows: A: {card_1} B: {card_2} C: {card_3} D: {card_4}. Which of the following best reflects your approach?"
        
        return mcq_options, prompt_template

def ensure_plot_dir(plot_dir):
    """Helper function to ensure plot directory exists"""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return plot_dir

def validate_model_availability(model_name, model_config):
    """Validate if a model is available (local) or needs to be downloaded (HuggingFace)"""
    if model_name not in model_config:
        available_models = list(model_config.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    model_info = model_config[model_name]
    
    if model_info['type'] == 'local':
        # Check if local model path exists
        if not os.path.exists(model_info['path']):
            raise FileNotFoundError(f"Local model not found at: {model_info['path']}")
        return True, "Local model available"
    
    elif model_info['type'] == 'huggingface':
        try:
            # Try to import transformers to check if it's available
            import transformers
            return True, "HuggingFace model (will be downloaded if not cached)"
        except ImportError:
            raise ImportError("transformers library not available. Install with: pip install transformers")
    
    return False, "Unknown model type"

def create_custom_model_config(custom_models=None):
    """Create a custom model configuration by extending the default"""
    manager = BiasDataManager(".")
    default_config = manager._get_default_model_config()
    
    if custom_models:
        default_config.update(custom_models)
    
    return default_config

class MultiModelExperimentRunner:
    """Runs bias experiments across multiple models"""
    
    def __init__(self, config_file=None, base_dir="."):
        self.base_dir = base_dir
        self.manager = BiasDataManager(base_dir, config_file=config_file)
        self.results = {}
    
    def run_single_model_experiment(self, model_name, bias_type, experiment_name, tokenizer, user_tag="USER: ", assistant_tag="ASSISTANT: ", testing=False):
        """Run a single experiment on a single model"""
        try:
            # Validate model availability
            validate_model_availability(model_name, self.manager.model_config)
            
            # Create training dataset
            dataset = self.manager.create_training_dataset(
                bias_type, tokenizer, user_tag, assistant_tag, testing=testing, model_name=model_name
            )
            
            # Load scenarios for testing
            max_scenarios = self.manager.get_max_scenarios()
            scenarios = self.manager.load_experiment_scenarios(
                bias_type, experiment_name, num_scenarios=max_scenarios, model_name=model_name
            )
            
            return {
                'model': model_name,
                'bias_type': bias_type,
                'experiment': experiment_name,
                'dataset_size': len(dataset['train']['data']),
                'test_scenarios': len(scenarios),
                'status': 'success',
                'dataset': dataset,
                'scenarios': scenarios
            }
            
        except Exception as e:
            return {
                'model': model_name,
                'bias_type': bias_type,
                'experiment': experiment_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_all_bias_experiments_single_model(self, model_name, tokenizer, user_tag="USER: ", assistant_tag="ASSISTANT: ", testing=False):
        """Run all bias experiments on a single model"""
        results = {}
        
        for bias_type in self.manager.get_all_bias_types():
            results[bias_type] = {}
            experiment_names = self.manager.get_experiment_names(bias_type, model_name)
            
            for experiment_name in experiment_names:
                print(f"Running {bias_type}.{experiment_name} on {model_name}...")
                result = self.run_single_model_experiment(
                    model_name, bias_type, experiment_name, tokenizer, user_tag, assistant_tag, testing
                )
                results[bias_type][experiment_name] = result
                
                if result['status'] == 'failed':
                    print(f"Failed: {result['error']}")
                else:
                    print(f"Success: {result['dataset_size']} training samples, {result['test_scenarios']} test scenarios")
        
        return results
    
    def run_all_models_all_experiments(self, tokenizer, user_tag="USER: ", assistant_tag="ASSISTANT: ", testing=False):
        """Run all experiments on all enabled models"""
        all_results = {}
        enabled_models = self.manager.get_enabled_models()
        
        print(f"Running experiments on {len(enabled_models)} enabled models...")
        
        for model_name in enabled_models.keys():
            print(f"\n=== Testing Model: {model_name} ===")
            model_results = self.run_all_bias_experiments_single_model(
                model_name, tokenizer, user_tag, assistant_tag, testing
            )
            all_results[model_name] = model_results
        
        self.results = all_results
        return all_results
    
    def get_model_comparison_summary(self):
        """Get summary comparing results across models"""
        if not self.results:
            return "No results available. Run experiments first."
        
        summary = {}
        
        for model_name, model_results in self.results.items():
            summary[model_name] = {
                'total_experiments': 0,
                'successful_experiments': 0,
                'failed_experiments': 0,
                'total_training_samples': 0,
                'total_test_scenarios': 0,
                'bias_types': {}
            }
            
            for bias_type, bias_results in model_results.items():
                summary[model_name]['bias_types'][bias_type] = {}
                
                for experiment_name, experiment_result in bias_results.items():
                    summary[model_name]['total_experiments'] += 1
                    
                    if experiment_result['status'] == 'success':
                        summary[model_name]['successful_experiments'] += 1
                        summary[model_name]['total_training_samples'] += experiment_result.get('dataset_size', 0)
                        summary[model_name]['total_test_scenarios'] += experiment_result.get('test_scenarios', 0)
                        summary[model_name]['bias_types'][bias_type][experiment_name] = 'success'
                    else:
                        summary[model_name]['failed_experiments'] += 1
                        summary[model_name]['bias_types'][bias_type][experiment_name] = 'failed'
        
        return summary
    
    def save_results(self, output_file):
        """Save experiment results to JSON file"""
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Also save summary
        summary_file = output_file.replace('.json', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(self.get_model_comparison_summary(), f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Summary saved to {summary_file}")


def load_scenarios_and_options(bias_type, experiment_name=None, num_scenarios=None):
    """
    Convenience function to load scenarios and MCQ options for API experiments.
    
    Args:
        bias_type: Type of bias ('authority', 'bandwagon', 'framing', 'confirmation')
        experiment_name: Name of the experiment (optional, will use first available)
        num_scenarios: Number of scenarios to load (optional, will load all)
        
    Returns:
        tuple: (scenarios, mcq_options, prompt_template)
    """
    # Determine the correct base directory based on current working directory
    current_dir = os.getcwd()
    if 'examples' in current_dir:
        # Running from examples/unified_bias directory
        base_dir = "."
    else:
        # Running from repository root (like api_bias.py)
        base_dir = os.path.join("examples", "unified_bias")
    
    # Initialize data manager
    data_manager = BiasDataManager(base_dir)
    
    # Get available experiment names for this bias type
    experiment_names = data_manager.get_experiment_names(bias_type)
    if not experiment_names:
        raise ValueError(f"No experiments found for bias type: {bias_type}")
    
    # Use provided experiment name or default to first available
    if experiment_name is None:
        experiment_name = experiment_names[0]
    elif experiment_name not in experiment_names:
        raise ValueError(f"Experiment '{experiment_name}' not found for bias type '{bias_type}'. Available: {experiment_names}")
    
    # Load scenarios
    scenarios = data_manager.load_experiment_scenarios(
        bias_type, experiment_name, num_scenarios=num_scenarios
    )
    
    # Get MCQ options and prompt template
    mcq_options, prompt_template = data_manager.get_mcq_options_and_templates(bias_type, experiment_name)
    
    return scenarios, mcq_options, prompt_template
