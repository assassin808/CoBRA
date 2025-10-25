import os
from typing import List, Optional

from control.env_config import BATCH_SIZE, print_config, NUM_PERMUTATIONS, TEMPERATURE
from control.experiment_utils import load_model_and_tags, evaluate_repread_accuracy
from control.repe_experiment import RepEControlExperiment
from control.prompt_experiment import PromptControlExperiment
from transformers import pipeline as hf_pipeline

from .utils_bias import BiasDataManager, ensure_plot_dir


def _setup_model_and_data(bias_type: str, model_name: Optional[str], is_testing_mode: bool):
    data_manager = BiasDataManager(os.path.dirname(__file__))

    if model_name:
        model_name_or_path = data_manager.get_model_path(bias_type, model_name)
        print(f"Using specified model: {model_name} -> {model_name_or_path}")
    else:
        model_name_or_path = data_manager.get_model_path(bias_type)
        print(f"Using default model: {model_name_or_path}")

    model, tokenizer, user_tag, assistant_tag, assistant_prompt_for_choice, device = load_model_and_tags(model_name_or_path)
    if model is None:
        raise RuntimeError("Model loading failed")

    # Load original scenarios
    num_scenarios = 4 if is_testing_mode else None
    experiment_names = data_manager.get_experiment_names(bias_type)
    all_experiments = {}
    for exp_name in experiment_names:
        scenarios = data_manager.load_experiment_scenarios(bias_type, exp_name, num_scenarios, model_name)
        mcq_options, prompt_template = data_manager.get_mcq_options_and_templates(bias_type, exp_name)
        if is_testing_mode and len(scenarios) > 4:
            scenarios = scenarios[:4]
        all_experiments[exp_name] = {
            'scenarios': scenarios,
            'mcq_options': mcq_options,
            'prompt_template': prompt_template
        }

    return data_manager, model, tokenizer, device, user_tag, assistant_tag, assistant_prompt_for_choice, all_experiments


def run_prompt_pipeline(
    bias_type: str,
    model_name: Optional[str] = None,
    is_testing_mode: bool = False,
    temperature: Optional[float] = None,
    likert_levels: Optional[List[int]] = None,
):
    print(f"\n=== PROMPT PIPELINE :: {bias_type} ===")
    data_manager, model, tokenizer, device, _, _, _, all_experiments = _setup_model_and_data(bias_type, model_name, is_testing_mode)

    effective_temperature = temperature if temperature is not None else TEMPERATURE
    temp_suffix = f"_temp{effective_temperature}" if temperature is not None else ""
    plot_dir_name = f"{bias_type}_plots{temp_suffix}"
    PLOT_DIR = ensure_plot_dir(plot_dir_name)

    levels = likert_levels if likert_levels is not None else list(range(0, 101, 5))

    prompt_experiment = PromptControlExperiment(
        model, tokenizer, device, PLOT_DIR,
        bias_type=bias_type,
        likert_levels=levels,
        is_testing_mode=is_testing_mode,
        num_permutations=NUM_PERMUTATIONS,
        reasoning_temperature=effective_temperature,
        choice_temperature=effective_temperature,
        experiment_suffix=temp_suffix,
    )

    for exp_name, exp_data in all_experiments.items():
        print(f"\n--- Prompt Control :: {bias_type.title()} - {exp_name.title()} ---")
        prompt_experiment.run(
            exp_data['scenarios'],
            exp_data['mcq_options'],
            exp_data['prompt_template'],
            f"{bias_type.title()} ({exp_name.title()})"
        )

    print(f"\n[Prompt] Results saved in {PLOT_DIR}")
    return True


def run_repe_pipeline(
    bias_type: str,
    model_name: Optional[str] = None,
    is_testing_mode: bool = False,
    temperature: Optional[float] = None,
    operators: Optional[List[str]] = None,
):
    print(f"\n=== RepE PIPELINE :: {bias_type} ===")
    data_manager, model, tokenizer, device, _, _, _, all_experiments = _setup_model_and_data(bias_type, model_name, is_testing_mode)

    # 1) RepReader setup + training data
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers - 1, -1))
    rep_reading_pipeline = hf_pipeline("rep-reading", model=model, tokenizer=tokenizer)

    print(f"Preparing generated dataset for RepReader...")
    dataset = data_manager.create_training_dataset(bias_type, tokenizer, user_tag="USER: ", assistant_tag="ASSISTANT: ", testing=is_testing_mode, model_name=model_name)
    if not dataset or not dataset['train']['data']:
        raise RuntimeError("Empty dataset for RepReader training")

    print("Training RepReader directions...")
    rep_reader = rep_reading_pipeline.get_directions(
        dataset['train']['data'],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        train_labels=dataset['train']['labels'],
        direction_method='pca',
        n_difference=1,
        batch_size=BATCH_SIZE,
        direction_finder_kwargs={'n_components': 4},
    )

    print("Evaluating RepReader accuracy...")
    temp_suffix = f"_temp{temperature}" if temperature is not None else ""
    plot_dir_name = f"{bias_type}_plots{temp_suffix}"
    PLOT_DIR = ensure_plot_dir(plot_dir_name)
    results, best_layer = evaluate_repread_accuracy(
        rep_reading_pipeline,
        dataset,
        rep_token,
        hidden_layers,
        rep_reader,
        PLOT_DIR,
        debug=is_testing_mode,
    )

    if not results or best_layer not in rep_reader.directions:
        print("[WARN] Best layer unavailable; skipping RepE control execution.")
        return False

    control_layers = sorted(rep_reader.directions.keys(), key=lambda l: results.get(l, 0), reverse=True)[:15]

    effective_temperature = temperature if temperature is not None else TEMPERATURE
    rep_control = hf_pipeline("rep-control", model=model, tokenizer=tokenizer, layers=control_layers, control_method="reading_vec")

    repe_experiment = RepEControlExperiment(
        model, tokenizer, device, PLOT_DIR,
        rep_control, rep_reader, control_layers,
        is_testing_mode=is_testing_mode, num_permutations=NUM_PERMUTATIONS,
        experiment_suffix=temp_suffix,
        reasoning_temperature=effective_temperature,
        choice_temperature=effective_temperature,
    )

    ops = operators if operators else ['linear_comb', 'projection']
    for exp_name, exp_data in all_experiments.items():
        print(f"\n--- RepE Control :: {bias_type.title()} - {exp_name.title()} ---")
        repe_experiment.run(
            exp_data['scenarios'],
            exp_data['mcq_options'],
            exp_data['prompt_template'],
            f"{bias_type.title()} ({exp_name.title()})",
            operators=ops,
        )

    print(f"\n[RepE] Results saved in {PLOT_DIR}")
    return True
