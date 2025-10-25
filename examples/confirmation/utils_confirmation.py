# utils_confirmation.py

import json
import random
import numpy as np

def create_confirmation_dataset(data_path: str, tokenizer, user_tag: str = "USER: ", assistant_tag: str = "ASSISTANT: ", seed: int = 42, testing: bool = False) -> dict:
    """
    CORRECTED VERSION:
    For each scenario, generate positive/negative pairs.
    The data is then structured into shuffled pairs for the RepReader,
    matching the format of the bandwagon dataset creation.
    """
    random.seed(seed)
    with open(data_path, 'r') as f:
        data = json.load(f)

    prompt_format = (
        "You are researching the topic: '{topic}'. "
        "Which of the following sources would you be most interested in reading?"
    )
    
    # Define persona templates for generating contrasting pairs  
    positive_persona_template = "Pretend you are very prone to confirmation bias, always seek only information that confirms your existing beliefs.\n\n{prompt}"
    neutral_persona_template = "{prompt}"  # Neutral state
    negative_persona_template = "Pretend you are not prone to confirmation bias, never seek only information that confirms your existing beliefs.\n\n{prompt}"
    
    # Define pairs to generate with custom ratio: 0.2 positive, 0.8 negative
    # We'll generate more negative pairs to achieve the desired ratio
    persona_pair_definitions = [
        (positive_persona_template, neutral_persona_template),  # 1 positive pair type
        (neutral_persona_template, negative_persona_template),  # 1 negative pair type
    ]

    all_generated_pairs = []

    for item in data:
        topic = item["Controversial Topic"]
        options = {
            "A": item["Supports Belief"],
            "B": item["Challenges Belief"],
            "C": item["Neutral Or Unrelated"]
        }
        options_string = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()])
        base_prompt = prompt_format.format(topic=topic) + options_string
        
        # Generate responses for each persona pair
        for preferred_template, rejected_template in persona_pair_definitions:
            preferred_user_prompt = preferred_template.format(prompt=base_prompt)
            rejected_user_prompt = rejected_template.format(prompt=base_prompt)
            
            # We need pairs for the RepReader. Let's create a pair for each answer choice.
            # This creates rich training data reflecting the persona's influence across different contexts.
            answer_choices = list(options.keys())
            for answer in answer_choices:
                # Create a pair for the empty assistant response (RepReader trains on this state)
                positive_examples = [f"{user_tag}{preferred_user_prompt}{assistant_tag}"]
                negative_examples = [f"{user_tag}{rejected_user_prompt}{assistant_tag}"]
                all_generated_pairs.extend([[pos, neg] for pos, neg in zip(positive_examples, negative_examples)])
                
                # Optionally, also create pairs for the full completion
                # This is what your original code did, so we'll keep it for consistency.
                positive_examples = [f"{user_tag}{preferred_user_prompt}{assistant_tag}{answer}"]
                negative_examples = [f"{user_tag}{rejected_user_prompt}{assistant_tag}{answer}"]
                all_generated_pairs.extend([[pos, neg] for pos, neg in zip(positive_examples, negative_examples)])

    if not all_generated_pairs:
        print("No examples were generated. Check data paths and processing logic.")
        return {'train': {'data': [], 'labels': []}, 'test': {'data': [], 'labels': []}}

    # --- CORRECTED PAIRING AND SHUFFLING LOGIC ---
    
    # 1. Create true pairs of [positive_example, negative_example]
    combined_data_true_pairs = all_generated_pairs
    random.shuffle(combined_data_true_pairs)
    
    # 2. Split pairs into train and test sets
    num_pairs = len(combined_data_true_pairs)
    ntrain_pairs = int(num_pairs * 0.6) if not testing else num_pairs
    if ntrain_pairs == 0 and num_pairs > 0: ntrain_pairs = 1

    train_pairs = combined_data_true_pairs[:ntrain_pairs]
    test_pairs = combined_data_true_pairs[ntrain_pairs:]

    # 3. Process TRAIN pairs into the format RepReader expects
    train_data_flat = []
    train_labels = []
    for pos_example, neg_example in train_pairs:
        # The true positive example is always the first in the original pair
        true_positive_example = pos_example
        
        shuffled_pair = [pos_example, neg_example]
        random.shuffle(shuffled_pair) # Shuffle the order within the pair
        
        train_data_flat.extend(shuffled_pair)
        # Create the label pair [1, 0] or [0, 1] based on the shuffled order
        train_labels.append([s == true_positive_example for s in shuffled_pair])

    # 4. Process TEST pairs similarly
    test_data_flat = []
    test_labels = []
    for pos_example, neg_example in test_pairs:
        true_positive_example = pos_example
        shuffled_pair = [pos_example, neg_example]
        random.shuffle(shuffled_pair)
        test_data_flat.extend(shuffled_pair)
        test_labels.append([s == true_positive_example for s in shuffled_pair])

    return {
        'train': {'data': train_data_flat, 'labels': train_labels}, 
        'test': {'data': test_data_flat, 'labels': test_labels}
    }


def create_confirmation_dataset_from_generated(data_path: str, tokenizer, user_tag: str = "USER: ", assistant_tag: str = "ASSISTANT: ", seed: int = 42, testing: bool = False, model_name: str = None) -> dict:
    """
    Processes generated confirmation bias data from the new generator format.
    This handles data generated by the scenario_generator.py and response_generator.py scripts.
    """
    random.seed(seed)
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Define persona templates for generating contrasting pairs

    positive_persona_template = "Pretend you have confirmation bias at level 100% out of 100% (where 0%=never seek only information that confirms your existing beliefs, 100%=always seek only information that confirms your existing beliefs). Each level increases bias by exactly 5%. Act with this precise level 100% of confirmation bias. \n\n{scenario_prompt}"
    neutral_persona_template = "{scenario_prompt}"  # Neutral state
    negative_persona_template = "Pretend you have confirmation bias at level 0% out of 100% (where 0%=not only seek information that confirms your existing beliefs, 100%=not only seek information that confirms your existing beliefs). Each level increases bias by exactly 5%. Act with this precise level 0% of confirmation bias. \n\n{scenario_prompt}"


    positive_persona_template = "Pretend you are very prone to confirmation bias, always seek the information that confirms your existing beliefs . \n\n{scenario_prompt}"
    neutral_persona_template = "{scenario_prompt}"  # Neutral state
    negative_persona_template = "Pretend you are not prone to confirmation bias, never seek only information that confirms your existing beliefs. \n\n{scenario_prompt}"
    
    # Define pairs to generate with custom ratio: 0.2 positive, 0.8 negative
    # We'll generate more negative pairs to achieve the desired ratio
    persona_pair_definitions = [
        (positive_persona_template, neutral_persona_template),  # 1 positive pair type
        (neutral_persona_template, negative_persona_template),  # 1 negative pair type
    ]

    all_generated_pairs = []
    
    for item in data:
        scenario_text = item.get("scenario", "")
        bias_type = item.get("bias_type", "confirmation")
        
        # Skip non-confirmation scenarios if this is specifically for confirmation bias
        if bias_type != "confirmation":
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
    ntrain_pairs = int(num_available_pairs * 0.6)
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


def load_confirmation_scenarios(data_path, num_scenarios=None):
    """
    Loads confirmation bias scenarios from a JSON file.
    Handles both 'BiasInfo' and 'Wason' style data.
    """
    # This function is fine, no changes needed here.
    with open(data_path, 'r') as f:
        data = json.load(f)

    if num_scenarios is not None:
        if len(data) > num_scenarios:
            data = random.sample(data, num_scenarios)

    scenarios = []
    for item in data:
        # Check if it's a Wason-style scenario
        if "Rule" in item:
            scenario = {
                "id": item.get("id"),
                "type": "wason_selection",
                "rule": item.get("Rule"),
                "card_1": item.get("Option A"),
                "card_2": item.get("Option B"),
                "card_3": item.get("Option C"),
                "card_4": item.get("Option D"),
            }
        # Assume it's a BiasInfo-style scenario otherwise
        else:
            scenario = {
                "id": item.get("id"),
                "type": "bias_info",
                "topic": item.get("Controversial Topic"),
                "options": {
                    "A": item.get("Supports Belief"),
                    "B": item.get("Challenges Belief"),
                    "C": item.get("Neutral Or Unrelated")
                }
            }
        scenarios.append(scenario)
        
    return scenarios