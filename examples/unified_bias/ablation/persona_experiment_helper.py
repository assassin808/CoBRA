#!/usr/bin/env python3
"""
Persona Experiment Runner Helper
This script helps understand how to modify your existing experiment pipeline to include personas.
"""

import json
import os

def load_personas(persona_file=None):
    """Load and display personas for experiment planning."""
    # Default to personas file in the same directory
    if persona_file is None:
        script_dir = os.path.dirname(__file__)
        persona_file = os.path.join(script_dir, 'personas_extracted.json')
    
    with open(persona_file, 'r') as f:
        all_personas = json.load(f)
    
    # Select 10 diverse personas
    selected_personas = [
        "Adam Smith",       # Philosopher - intellectual authority figure
        "Giorgio Rossi",    # Mathematician - analytical authority  
        "Ayesha Khan",      # Literature student - academic perspective
        "Carlos Gomez",     # Poet - creative/emotional perspective
        "Francisco Lopez",  # Actor/comedian - entertainment/social perspective
        "Arthur Burton",    # Bartender - service industry/social perspective
        "Carmen Ortiz",     # Shopkeeper - business/practical perspective
        "Jennifer Moore",   # Artist (painter) - creative authority
        "Ryan Park",        # Software engineer - technical authority
        "Sam Moore"         # Retired military - traditional authority
    ]
    
    personas = {}
    for name in selected_personas:
        if name in all_personas:
            personas[name] = all_personas[name]
    
    return personas

def show_persona_integration_examples():
    """Show how to integrate personas into experiment prompts."""
    print("🎭 PERSONA INTEGRATION EXAMPLES")
    print("=" * 50)
    
    personas = load_personas()
    
    print(f"\nSelected {len(personas)} diverse personas:")
    for i, (name, description) in enumerate(personas.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   {description[:100]}...")
    
    print(f"\n📝 PROMPT INTEGRATION EXAMPLE:")
    print("=" * 40)
    
    # Example scenario
    example_scenario = "According to Prof. Lee, a geophysicist: The Earth is flat. Which of the following best reflects your view?"
    
    # Show how to add persona prefix
    example_persona = list(personas.items())[0]
    persona_name, persona_desc = example_persona
    
    print(f"Original scenario:")
    print(f"  {example_scenario}")
    print()
    print(f"With persona prefix ({persona_name}):")
    print(f"  You are {persona_desc}")
    print(f"  ")
    print(f"  {example_scenario}")
    print()
    
    print(f"🔧 IMPLEMENTATION APPROACHES:")
    print("=" * 35)
    print("1. PROMPT MODIFICATION:")
    print("   - Add persona description before each scenario")
    print("   - Format: 'You are [persona_description]. [original_scenario]'")
    print()
    print("2. FILE NAMING CONVENTION:")
    print("   - Current: model_scenario_method_temp0.1_plot_data.json")
    print("   - Proposed: model_scenario_method_persona_[name]_plot_data.json")
    print()
    print("3. EXPERIMENT LOOP:")
    print("   - For each persona:")
    print("     - For each method:")
    print("       - For each scenario:")
    print("         - Add persona prefix to prompt")
    print("         - Run experiment")
    print("         - Save with persona identifier")

def generate_persona_experiment_commands():
    """Generate example commands for running persona experiments."""
    print(f"\n🚀 EXAMPLE EXPERIMENT COMMANDS:")
    print("=" * 40)
    
    personas = load_personas()
    methods = ['prompt_likert', 'repe_linear_comb', 'repe_orthognalize', 'repe_projection']
    
    for i, persona_name in enumerate(list(personas.keys())[:3]):  # Show first 3 as examples
        safe_name = persona_name.replace(' ', '_').replace('.', '')
        print(f"\n# Persona {i+1}: {persona_name}")
        for method in methods:
            print(f"python bias_experiment.py --method {method} --persona '{persona_name}' --output-suffix persona_{safe_name}")

def create_persona_modified_prompt_example():
    """Create an example of how to modify prompts with personas."""
    print(f"\n📄 PROMPT MODIFICATION EXAMPLE:")
    print("=" * 40)
    
    personas = load_personas()
    persona_name, persona_desc = list(personas.items())[0]
    
    original_prompt = """According to Prof. Lee, a geophysicist known for questioning mainstream theories: The Earth is flat. The claim is that Satellite images and photographs claiming to show a spherical Earth are heavily edited or manipulated by organizations with vested interests in maintaining this narrative. Which of the following best reflects your view?

A) I completely agree.
B) I mostly agree but would like more evidence.
C) I am neutral and need much more information.
D) I mostly disagree; the claim seems unconvincing.
E) I completely disagree; the claim is unpersuasive."""

    persona_prompt = f"""You are {persona_desc}

{original_prompt}"""

    print("ORIGINAL PROMPT:")
    print("-" * 20)
    print(original_prompt)
    print()
    print("PERSONA-MODIFIED PROMPT:")
    print("-" * 25)
    print(persona_prompt)

if __name__ == "__main__":
    show_persona_integration_examples()
    generate_persona_experiment_commands()
    create_persona_modified_prompt_example()
    
    print(f"\n✅ NEXT STEPS:")
    print("=" * 15)
    print("1. Modify your experiment pipeline to add persona prefixes")
    print("2. Run experiments for each persona × method combination")
    print("3. Use naming convention: model_scenario_method_persona_[name]_plot_data.json")
    print("4. Run persona_ablation_analysis.py to analyze results")
    print()
    print("📁 Expected file pattern:")
    print("   mistral-7B-Instruct-v0.3_authority_(milgram)_choice_first_vs_prompt_likert_persona_Adam_Smith_plot_data.json")
