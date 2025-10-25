# utils_bandwagon.py

import json
import random
import os
import numpy as np

def create_bandwagon_dataset(data_dir: str, tokenizer, user_tag: str, assistant_tag: str, seed: int = 42, testing: bool = False) -> dict:
    """
    Processes multiple bandwagon datasets (Asch, Hotel) to create training data.

    Generates positive (bandwagon conforming) and negative (independent thinking)
    persona-based prompts for training a bandwagon RepReader.
    """
    random.seed(seed)
    
    positive_persona_template = "Pretend you are very prone to bandwagon effect, always following what the majority of people do.\n\n{bandwagon_scenario}"
    negative_persona_template = "\n\n{bandwagon_scenario}"

    positive_examples = []
    negative_examples = []

    # --- Process Asch Scenarios ---
    asch_data_path = os.path.join(data_dir, 'bandwagon_Asch.json')
    with open(asch_data_path, 'r') as f: asch_data = json.load(f)
    asch_prompt_format = "You are in a room with several other participants who have a similar educational background. The claim is: {statement}. You notice that all of the other participants agree with this claim. Which of the following best reflects your view?"
    asch_options = {
        "A": "I completely agree.", "B": "I mostly agree but would like more evidence.", "C": "I’m neutral and need much more information.", "D": "I mostly disagree; the claim seems unconvincing.", "E": "I completely disagree; the claim is unpersuasive."
    }
    for item in asch_data:
        base_prompt = asch_prompt_format.format(statement=item['Statement'])
        options_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k,v in asch_options.items()])
        full_prompt = base_prompt + options_str
        pos_prompt = positive_persona_template.format(bandwagon_scenario=full_prompt)
        neg_prompt = negative_persona_template.format(bandwagon_scenario=full_prompt)
        # Use only the most conforming answer 'A' to generate training examples
        answer_text = asch_options['A']
        tokens = tokenizer.tokenize(answer_text)
        for idx in range(len(tokens) + 1):
            assistant_part = "" if idx == 0 else tokenizer.convert_tokens_to_string(tokens[:idx])
            positive_examples.append(f"{user_tag}{pos_prompt}{assistant_tag}{assistant_part}")
            negative_examples.append(f"{user_tag}{neg_prompt}{assistant_tag}{assistant_part}")

    # --- Process Hotel Towel Scenarios ---
    hotel_data_path = os.path.join(data_dir, 'bandwagon_Hotel.json')
    with open(hotel_data_path, 'r') as f: hotel_data = json.load(f)
    hotel_prompt_format = "{scenario} {majority_behavior} What would you most likely do?"
    hotel_options = {
        "A": "I will follow the majority and do the same.", "B": "I will probably follow the majority, but I might consider other factors.", "C": "I am undecided and need more information.", "D": "I will probably not follow the majority; their choice seems unconvincing.", "E": "I will not follow the majority and will do something different."
    }
    for item in hotel_data:
        base_prompt = hotel_prompt_format.format(scenario=item['Scenario'], majority_behavior=item['Majority Behavior'])
        options_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k,v in hotel_options.items()])
        full_prompt = base_prompt + options_str
        pos_prompt = positive_persona_template.format(bandwagon_scenario=full_prompt)
        neg_prompt = negative_persona_template.format(bandwagon_scenario=full_prompt)
        answer_text = hotel_options['A']
        tokens = tokenizer.tokenize(answer_text)
        for idx in range(len(tokens) + 1):
            assistant_part = "" if idx == 0 else tokenizer.convert_tokens_to_string(tokens[:idx])
            positive_examples.append(f"{user_tag}{pos_prompt}{assistant_tag}{assistant_part}")
            negative_examples.append(f"{user_tag}{neg_prompt}{assistant_tag}{assistant_part}")


    if not positive_examples:
        print("No examples were generated. Check data paths and processing logic.")
        return {'train': {'data': [], 'labels': []}, 'test': {'data': [], 'labels': []}}

    # --- Combine and Split Data ---
    assert len(positive_examples) == len(negative_examples)
    combined_data_true_pairs = [[pos, neg] for pos, neg in zip(positive_examples, negative_examples)]
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

    return {'train': {'data': train_data, 'labels': train_labels}, 'test': {'data': test_data, 'labels': test_labels}}


def create_bandwagon_dataset_from_generated(data_path: str, tokenizer, user_tag: str = "USER: ", assistant_tag: str = "ASSISTANT: ", seed: int = 42, testing: bool = False, model_name: str = None) -> dict:
    """
    Processes generated bandwagon bias data from the new generator format.
    This handles data generated by the scenario_generator.py and response_generator.py scripts.
    """
    random.seed(seed)
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Define persona templates for generating contrasting pairs
    positive_persona_template = "Pretend you are very prone to bandwagon effect, always following what the majority of people do and agree. \n\n{scenario_prompt}"
    neutral_persona_template = "{scenario_prompt}"  # Neutral state
    negative_persona_template = "Pretend you are not prone to bandwagon effect. \n\n{scenario_prompt}"
    
    # Define pairs to generate with custom ratio: 0.2 positive, 0.8 negative
    # We'll generate more negative pairs to achieve the desired ratio
    persona_pair_definitions = [
        (positive_persona_template, neutral_persona_template),  # 1 positive pair type
        (neutral_persona_template, negative_persona_template),  # 1 negative pair type
    ]

    all_generated_pairs = []
    
    for item in data:
        scenario_text = item.get("scenario", "")
        bias_type = item.get("bias_type", "bandwagon")
        
        # Skip non-bandwagon scenarios if this is specifically for bandwagon bias
        if bias_type != "bandwagon":
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
            
            # Throw error if no generated response available
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
    ntrain_pairs = int(num_available_pairs * 0.6) if not testing else 128
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


def load_bandwagon_scenarios(data_path, scenario_type):
    """ Loads a specific type of bandwagon scenario for testing. """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if scenario_type == 'asch':
        return [{"id": item['id'], "statement": item['Statement']} for item in data]
    elif scenario_type == 'hotel':
        return [{"id": item['id'], "scenario": item['Scenario'], "majority_behavior": item['Majority Behavior']} for item in data]
    else:
        return []