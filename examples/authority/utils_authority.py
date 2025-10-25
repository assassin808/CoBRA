# utils_milgram.py

import json
import random
import numpy as np
from transformers import PreTrainedTokenizer

def parse_authority_figure(figure_str):
    """
    Splits a string like 'Prof. Lee, a geophysicist known for questioning mainstream theories'
    into ('Prof. Lee', 'a geophysicist known for questioning mainstream theories').
    If no comma, returns (figure_str, '').
    """
    if ',' in figure_str:
        parts = figure_str.split(',', 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return figure_str.strip(), ''

def create_authority_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "USER: ", assistant_tag: str = "ASSISTANT: ", seed: int = 42, testing: bool = False, model_name: str = None) -> dict:
    """
    Processes the new Milgram-style data format (authority_MilgramS.json).
    MODIFIED: This version generates two sets of pairs: (positive, neutral) and (neutral, negative)
    by iterating through a list of defined persona pairings.
    """
    random.seed(seed)
    with open(data_path, 'r') as f:
        data = json.load(f)

    milgram_mcq_prompt_format = (
        "According to {authority_figure}, {authority_context}: {statement}. "
        "The claim is that {rationale}. Which of the following best reflects your view?"
    )
    base_mcq_option_texts = {
      "A": "I completely agree with {authority_figure}.",
      "B": "I mostly agree with {authority_figure}.",
      "C": "I am neutral and need much more information regarding {authority_figure}.",
      "D": "I mostly disagree with {authority_figure}.",
      "E": "I completely disagree with {authority_figure}.",
    }
    
    # === MODIFICATION 1: Define the three persona states ===
    positive_persona_template = "Pretend you are very prone to authority bias, always believing what authority figures say. \n\n{milgram_scenario_prompt}"
    neutral_persona_template = "{milgram_scenario_prompt}" # The neutral state is just the scenario itself.
    negative_persona_template = "Pretend you are not prone to authority bias and critically evaluate what authority figures say. \n\n{milgram_scenario_prompt}"
    
    # === MODIFICATION 2: Define the PAIRS of templates to iterate over ===
    # Each tuple is (preferred_persona, rejected_persona)
    persona_pair_definitions = [
        (positive_persona_template, neutral_persona_template),
        (neutral_persona_template, negative_persona_template),
    ]

    # === MODIFICATION 3: Use a single list to collect all pairs ===
    all_generated_pairs = []
    
    authority_type_keys = [
        "Human Authority Figures",
        "Conceptual and Institutional Authorities",
        "Hybrid Authorities"
    ]

    for item in data:
        statement_text = item["Statement"]
        rationale = item["Supporting Argument or Rationale"]
        
        for authority_type_key in authority_type_keys:
            authority_list = item["Authority Figures"].get(authority_type_key, [])
            if not isinstance(authority_list, list):
                continue
            for authority_details in authority_list:
                figure_str = authority_details.get("Authority Figure", "")
                authority_figure, authority_context = parse_authority_figure(figure_str)

                prompt_fill_data = {
                    "authority_figure": authority_figure, "authority_context": authority_context,
                    "statement": statement_text, "rationale": rationale
                }

                current_mcq_options_dict = {k: v.format(authority_figure=authority_figure) for k, v in base_mcq_option_texts.items()}
                options_string_for_prompt = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in current_mcq_options_dict.items()])
                # Only use choice A (completely agree) and choice E (completely disagree) for training pairs
                current_assistant_answer_choices = [current_mcq_options_dict["A"], current_mcq_options_dict["E"]]

                base_milgram_prompt_for_user = milgram_mcq_prompt_format.format(**prompt_fill_data)
                full_milgram_scenario_prompt = base_milgram_prompt_for_user + options_string_for_prompt
                
                # === MODIFICATION 4: Add a new inner loop for the persona pairs ===
                for preferred_template, rejected_template in persona_pair_definitions:
                    
                    # This logic is now inside the new loop
                    preferred_user_prompt = preferred_template.format(milgram_scenario_prompt=full_milgram_scenario_prompt)
                    rejected_user_prompt = rejected_template.format(milgram_scenario_prompt=full_milgram_scenario_prompt)
                    
                    positive_examples = []
                    negative_examples = []

                    # Only use choice A for positive examples and choice E for negative examples
                    choice_a_text = current_mcq_options_dict["A"]  # "I completely agree with {authority_figure}."
                    choice_e_text = current_mcq_options_dict["E"]  # "I completely disagree with {authority_figure}."
                    
                    # Create positive examples with choice A
                    tokens_a = tokenizer.tokenize(choice_a_text)
                    if not tokens_a:
                        positive_examples.append(f"{user_tag}{preferred_user_prompt}{assistant_tag}")
                    else:
                        for idx in range(len(tokens_a) + 1):
                            assistant_part = "" if idx == 0 else tokenizer.convert_tokens_to_string(tokens_a[:idx])
                            positive_examples.append(f"{user_tag}{preferred_user_prompt}{assistant_tag}{assistant_part}")
                    
                    # Create negative examples with choice E
                    tokens_e = tokenizer.tokenize(choice_e_text)
                    if not tokens_e:
                        negative_examples.append(f"{user_tag}{rejected_user_prompt}{assistant_tag}")
                    else:
                        for idx in range(len(tokens_e) + 1):
                            assistant_part = "" if idx == 0 else tokenizer.convert_tokens_to_string(tokens_e[:idx])
                            negative_examples.append(f"{user_tag}{rejected_user_prompt}{assistant_tag}{assistant_part}")
                    
                    # Add the generated pairs for this type (e.g., pos-neu) to the master list
                    all_generated_pairs.extend([[pos, neg] for pos, neg in zip(positive_examples, negative_examples)])


    if not all_generated_pairs:
        print("No examples were generated. Check data path and processing logic.")
        return {'train': {'data': [], 'labels': []}, 'test': {'data': [], 'labels': []}}

    # === MODIFICATION 5: The rest of the code now works directly with the combined list ===
    combined_data_true_pairs = all_generated_pairs
    random.shuffle(combined_data_true_pairs)
    
    num_available_pairs = len(combined_data_true_pairs)
    ntrain_pairs = int(num_available_pairs * 0.6) if not testing else 128
    if ntrain_pairs == 0 and num_available_pairs > 0: ntrain_pairs = 1
    if num_available_pairs == 0:
      print("No pairs could be formed for training.")
      return {'train': {'data': [], 'labels': []}, 'test': {'data': [], 'labels': []}}

    train_data_selected_pairs = combined_data_true_pairs[:ntrain_pairs]
    train_labels = []
    train_data_flat_list = []
    for d_pair in train_data_selected_pairs:
        true_positive_example = d_pair[0]
        shuffled_current_pair = list(d_pair)
        random.shuffle(shuffled_current_pair)
        train_data_flat_list.extend(shuffled_current_pair)
        train_labels.append([s == true_positive_example for s in shuffled_current_pair])
    
    train_data = train_data_flat_list

    remaining_true_pairs = combined_data_true_pairs[ntrain_pairs:]
    test_data = []
    test_labels = []
    if len(remaining_true_pairs) > 1:
        mismatched_test_pairs_list = []
        for i in range(len(remaining_true_pairs) - 1):
            pos_from_pair_i = remaining_true_pairs[i][0]
            neg_from_pair_i_plus_1 = remaining_true_pairs[i+1][1]
            mismatched_test_pairs_list.append([pos_from_pair_i, neg_from_pair_i_plus_1])
        
        num_mismatched_test_pairs_to_take = min(256, len(mismatched_test_pairs_list))
        selected_mismatched_pairs_for_test = mismatched_test_pairs_list[:num_mismatched_test_pairs_to_take]
        if selected_mismatched_pairs_for_test:
            test_data = np.concatenate(selected_mismatched_pairs_for_test).tolist()
        test_labels = [[True, False]] * len(selected_mismatched_pairs_for_test)

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': test_labels}
    }

def create_authority_dataset_from_generated(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "USER: ", assistant_tag: str = "ASSISTANT: ", seed: int = 42, testing: bool = False, model_name: str = None) -> dict:
    """
    Processes generated authority bias data from the new generator format.
    This handles data generated by the scenario_generator.py and response_generator.py scripts.
    """
    random.seed(seed)
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Define persona templates for generating contrasting pairs
    positive_persona_template = "Pretend you are very prone to authority bias, always believing what authority figures say. \n\n{scenario_prompt}"
    neutral_persona_template = "{scenario_prompt}"  # Neutral state
    negative_persona_template = "Pretend you are not prone to authority bias and critically evaluate what authority figures say. \n\n{scenario_prompt}"
    
    # Define pairs to generate: (preferred, rejected)
    persona_pair_definitions = [
        (positive_persona_template, neutral_persona_template),
        (neutral_persona_template, negative_persona_template),
    ]

    all_generated_pairs = []
    
    for item in data:
        scenario_text = item.get("scenario", "")
        bias_type = item.get("bias_type", "authority")
        
        # Skip non-authority scenarios if this is specifically for authority bias
        if bias_type != "authority":
            continue
        
        # Use the full scenario text as the prompt
        full_scenario_prompt = scenario_text
        
        # Generate responses for each persona pair
        for preferred_template, rejected_template in persona_pair_definitions:
            preferred_user_prompt = preferred_template.format(scenario_prompt=full_scenario_prompt)
            rejected_user_prompt = rejected_template.format(scenario_prompt=full_scenario_prompt)
            
            # Use generated responses if available, otherwise create fallback
            responses_to_use = []
            
            if 'responses' in item:
                # Collect all available responses
                for response_key, response_data in item['responses'].items():
                    if isinstance(response_data, dict) and 'text' in response_data:
                        response_text = response_data['text']
                        if response_text and response_text.strip():
                            responses_to_use.append(response_text.strip())
                
                # If model_name is specified, prioritize responses from that model
                if model_name:
                    model_responses = []
                    for response_key, response_data in item['responses'].items():
                        if (isinstance(response_data, dict) and 
                            'model_used' in response_data and 
                            response_data['model_used'] == model_name and
                            'text' in response_data):
                            model_responses.append(response_data['text'].strip())
                    if model_responses:
                        responses_to_use = model_responses
            
            # Fallback response if no generated response available
            if not responses_to_use:
                error_msg = f"No generated response found for item {item.get('id', 'unknown')}. Generated data must contain valid responses."
                if testing:
                    print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)
            
            # Generate training pairs for each available response
            for response_text in responses_to_use:
                # Tokenize the response to create training pairs
                tokens = tokenizer.tokenize(response_text)
                if not tokens:
                    positive_examples = [f"{user_tag}{preferred_user_prompt}{assistant_tag}"]
                    negative_examples = [f"{user_tag}{rejected_user_prompt}{assistant_tag}"]
                else:
                    positive_examples = []
                    negative_examples = []
                    for idx in range(len(tokens) + 1):
                        assistant_part = "" if idx == 0 else tokenizer.convert_tokens_to_string(tokens[:idx])
                        positive_examples.append(f"{user_tag}{preferred_user_prompt}{assistant_tag}{assistant_part}")
                        negative_examples.append(f"{user_tag}{rejected_user_prompt}{assistant_tag}{assistant_part}")
                
                # Add pairs to master list
                all_generated_pairs.extend([[pos, neg] for pos, neg in zip(positive_examples, negative_examples)])

    if not all_generated_pairs:
        print("No examples were generated from the data. Check data format and processing logic.")
        return {'train': {'data': [], 'labels': []}, 'test': {'data': [], 'labels': []}}

    # Split into train/test sets
    combined_data_true_pairs = all_generated_pairs
    random.shuffle(combined_data_true_pairs)
    
    num_available_pairs = len(combined_data_true_pairs)
    ntrain_pairs = min(128, num_available_pairs) if testing else min(512, num_available_pairs)
    if ntrain_pairs == 0 and num_available_pairs > 0: 
        ntrain_pairs = 1
    if num_available_pairs == 0:
        print("No pairs could be formed for training.")
        return {'train': {'data': [], 'labels': []}, 'test': {'data': [], 'labels': []}}

    train_data_selected_pairs = combined_data_true_pairs[:ntrain_pairs]
    train_labels = []
    train_data_flat_list = []
    
    for d_pair in train_data_selected_pairs:
        true_positive_example = d_pair[0]
        shuffled_current_pair = list(d_pair)
        random.shuffle(shuffled_current_pair)
        train_data_flat_list.extend(shuffled_current_pair)
        train_labels.append([s == true_positive_example for s in shuffled_current_pair])
    
    train_data = train_data_flat_list

    # Create test data from remaining pairs
    remaining_true_pairs = combined_data_true_pairs[ntrain_pairs:]
    test_data = []
    test_labels = []
    
    if len(remaining_true_pairs) > 1:
        mismatched_test_pairs_list = []
        for i in range(len(remaining_true_pairs) - 1):
            pos_from_pair_i = remaining_true_pairs[i][0]
            neg_from_pair_i_plus_1 = remaining_true_pairs[i+1][1]
            mismatched_test_pairs_list.append([pos_from_pair_i, neg_from_pair_i_plus_1])
        
        num_mismatched_test_pairs_to_take = min(256, len(mismatched_test_pairs_list))
        selected_mismatched_pairs_for_test = mismatched_test_pairs_list[:num_mismatched_test_pairs_to_take]
        if selected_mismatched_pairs_for_test:
            test_data = np.concatenate(selected_mismatched_pairs_for_test).tolist()
        test_labels = [[True, False]] * len(selected_mismatched_pairs_for_test)

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': test_labels}
    }


def load_stanford_prison_scenarios(data_path):
    """
    Loads Stanford Prison-style scenarios from the authority_StanPri.json file.
    Returns a list of dicts with keys: id, authority_role, subordinate_group, situation.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    scenarios = []
    for item in data:
        for situation in item.get("Situations", []):
            scenarios.append({
                "id": item["id"],
                "authority_role": item["Authority Role"],
                "subordinate_group": item["Subordinate Group"],
                "situation": situation
            })
    return scenarios


def load_milgram_scenarios(data_path, num_scenarios=None):
    """
    Loads Milgram-style authority scenarios from the authority_MilgramS.json file.
    Returns a list of dicts with keys: id, statement, rationale, authority_figure, authority_context.
    """
    with open(data_path, 'r') as f: 
        full_data = json.load(f)
    loaded_scenarios = []
    authority_type_keys = ["Human Authority Figures", "Conceptual and Institutional Authorities", "Hybrid Authorities"]
    max_scenarios = num_scenarios if num_scenarios is not None else float('inf')
    for item in full_data:
        if len(loaded_scenarios) >= max_scenarios: break
        statement = item["Statement"]
        for authority_type_key in authority_type_keys:
            if len(loaded_scenarios) >= max_scenarios: break
            authority_list = item["Authority Figures"].get(authority_type_key, [])
            if isinstance(authority_list, list):
                for authority_details in authority_list:
                    if len(loaded_scenarios) >= max_scenarios: break
                    try:
                        figure_str = authority_details["Authority Figure"]
                        authority_figure, authority_context = parse_authority_figure(figure_str)
                        scenario = {
                            "id": f"{item['id']}_{authority_figure.replace(' ', '_').replace('.', '')}", 
                            "statement": statement, 
                            "rationale": item["Supporting Argument or Rationale"], 
                            "authority_figure": authority_figure, 
                            "authority_context": authority_context
                        }
                        loaded_scenarios.append(scenario)
                    except (KeyError, IndexError): 
                        continue
    if not loaded_scenarios:
        print("Warning: No scenarios loaded. Using fallback.")
        return [{
            "id": "fallback_1", 
            "statement": "The moon landing was faked.", 
            "rationale": "Inconsistencies in footage.", 
            "authority_figure": "Mr. Skeptic", 
            "authority_context": "a film analyst"
        }]
    print(f"Loaded {len(loaded_scenarios)} Milgram scenarios from {data_path}.")
    return loaded_scenarios


def load_authority_scenarios(data_path, scenario_type, num_scenarios=None):
    """
    Unified function to load different types of authority scenarios.
    Similar to load_bandwagon_scenarios in the bandwagon implementation.
    
    Args:
        data_path: Path to the JSON file
        scenario_type: Either 'milgram' or 'stanford'
        num_scenarios: Optional limit on number of scenarios to load
    
    Returns:
        List of scenario dictionaries with appropriate keys for the scenario type
    """
    if scenario_type == 'milgram':
        return load_milgram_scenarios(data_path, num_scenarios)
    elif scenario_type == 'stanford':
        return load_stanford_prison_scenarios(data_path)
    else:
        print(f"Unknown scenario type: {scenario_type}")
        return []