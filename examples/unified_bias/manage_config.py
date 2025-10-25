#!/usr/bin/env python3
"""
Model Configuration Manager

This script helps manage the model configuration file.

Usage:
    python manage_config.py [--config model_config.json] [command] [options]
    
Commands:
    list - List all models
    enable MODEL_NAME - Enable a model
    disable MODEL_NAME - Disable a model
    add MODEL_NAME PATH TYPE DESCRIPTION - Add a new model
    set-default MODEL_NAME - Set default model
    validate - Validate all models
"""

import argparse
import json
import os
import sys

def load_config(config_file):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {config_file}: {e}")
        sys.exit(1)

def save_config(config, config_file):
    """Save configuration to file"""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_file}")

def list_models(config):
    """List all models in configuration"""
    models = config.get('models', {})
    print(f"Found {len(models)} models:")
    print(f"Default model: {config.get('default_model', 'Not set')}")
    print()
    
    for name, info in models.items():
        status = "enabled" if info.get('enabled', True) else "disabled"
        print(f"  {name}:")
        print(f"    Type: {info['type']}")
        print(f"    Path: {info['path']}")
        print(f"    Description: {info['description']}")
        print(f"    Status: {status}")
        print()

def enable_model(config, model_name):
    """Enable a model"""
    if model_name not in config.get('models', {}):
        print(f"Model {model_name} not found in configuration")
        return False
    
    config['models'][model_name]['enabled'] = True
    print(f"Model {model_name} enabled")
    return True

def disable_model(config, model_name):
    """Disable a model"""
    if model_name not in config.get('models', {}):
        print(f"Model {model_name} not found in configuration")
        return False
    
    config['models'][model_name]['enabled'] = False
    print(f"Model {model_name} disabled")
    return True

def add_model(config, model_name, path, model_type, description):
    """Add a new model to configuration"""
    if 'models' not in config:
        config['models'] = {}
    
    if model_type not in ['local', 'huggingface']:
        print(f"Invalid model type: {model_type}. Must be 'local' or 'huggingface'")
        return False
    
    config['models'][model_name] = {
        'path': path,
        'type': model_type,
        'description': description,
        'enabled': True
    }
    
    print(f"Model {model_name} added to configuration")
    return True

def set_default_model(config, model_name):
    """Set default model"""
    if model_name not in config.get('models', {}):
        print(f"Model {model_name} not found in configuration")
        return False
    
    config['default_model'] = model_name
    print(f"Default model set to {model_name}")
    return True

def validate_models(config):
    """Validate all models in configuration"""
    # Add utils_bias import
    sys.path.append(os.path.dirname(__file__))
    from utils_bias import validate_model_availability
    
    models = config.get('models', {})
    print(f"Validating {len(models)} models...")
    
    for model_name, model_info in models.items():
        if not model_info.get('enabled', True):
            print(f"  {model_name}: skipped (disabled)")
            continue
            
        try:
            is_valid, message = validate_model_availability(model_name, models)
            print(f"  {model_name}: ✓ {message}")
        except Exception as e:
            print(f"  {model_name}: ✗ {e}")

def main():
    parser = argparse.ArgumentParser(description='Manage model configuration')
    parser.add_argument('--config', default='model_config.json',
                       help='Configuration file path (default: model_config.json)')
    parser.add_argument('command', nargs='?', default='list',
                       help='Command to execute (list, enable, disable, add, set-default, validate)')
    parser.add_argument('args', nargs='*',
                       help='Arguments for the command')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config_changed = False
    
    # Execute command
    if args.command == 'list':
        list_models(config)
    
    elif args.command == 'enable':
        if not args.args:
            print("Usage: enable MODEL_NAME")
            sys.exit(1)
        config_changed = enable_model(config, args.args[0])
    
    elif args.command == 'disable':
        if not args.args:
            print("Usage: disable MODEL_NAME")
            sys.exit(1)
        config_changed = disable_model(config, args.args[0])
    
    elif args.command == 'add':
        if len(args.args) < 4:
            print("Usage: add MODEL_NAME PATH TYPE DESCRIPTION")
            sys.exit(1)
        config_changed = add_model(config, args.args[0], args.args[1], args.args[2], ' '.join(args.args[3:]))
    
    elif args.command == 'set-default':
        if not args.args:
            print("Usage: set-default MODEL_NAME")
            sys.exit(1)
        config_changed = set_default_model(config, args.args[0])
    
    elif args.command == 'validate':
        validate_models(config)
    
    else:
        print(f"Unknown command: {args.command}")
        print("Available commands: list, enable, disable, add, set-default, validate")
        sys.exit(1)
    
    # Save configuration if changed
    if config_changed:
        save_config(config, args.config)

if __name__ == '__main__':
    main()
