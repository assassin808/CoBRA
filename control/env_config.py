"""
Environment configuration loader for Authority bias experiments
"""
import os
from dotenv import load_dotenv

# Load .env file from the root of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

def get_env_int(key, default):
    """Get an integer environment variable with fallback to default"""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def get_env_float(key, default):
    """Get a float environment variable with fallback to default"""
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def get_env_str(key, default):
    """Get a string environment variable with fallback to default"""
    return os.getenv(key, default)

# Batch size settings
BATCH_SIZE = get_env_int('BATCH_SIZE', 1)
REASONING_BATCH_SIZE = get_env_int('REASONING_BATCH_SIZE', 1)
REP_READING_BATCH_SIZE = get_env_int('REP_READING_BATCH_SIZE', 8)
BANDWAGON_BATCH_SIZE = get_env_int('BANDWAGON_BATCH_SIZE', 16)

# HuggingFace settings
HF_TOKEN = get_env_str('HF_TOKEN', None)

# OpenRouter API settings
OPENROUTER_API_KEY = get_env_str('OPENROUTER_API_KEY', None)

# Model settings
DEFAULT_MODEL = get_env_str('DEFAULT_MODEL', 'mistral-7b-local')

# Experiment settings
NUM_PERMUTATIONS = get_env_int('NUM_PERMUTATIONS', 1)
MAX_NEW_TOKENS = get_env_int('MAX_NEW_TOKENS', 128)
TEMPERATURE = get_env_float('TEMPERATURE', 1.0)
STEP_SIZE = get_env_float('STEP_SIZE', 1.0)

# Print loaded configuration (for debugging)
def print_config():
    """Print the loaded configuration"""
    print("="*50)
    print("LOADED ENVIRONMENT CONFIGURATION")
    print("="*50)
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"REASONING_BATCH_SIZE: {REASONING_BATCH_SIZE}")
    print(f"REP_READING_BATCH_SIZE: {REP_READING_BATCH_SIZE}")
    print(f"BANDWAGON_BATCH_SIZE: {BANDWAGON_BATCH_SIZE}")
    print(f"HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'}")
    print(f"OPENROUTER_API_KEY: {'SET' if OPENROUTER_API_KEY else 'NOT SET'}")
    print(f"DEFAULT_MODEL: {DEFAULT_MODEL}")
    print(f"NUM_PERMUTATIONS: {NUM_PERMUTATIONS}")
    print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print(f"STEP_SIZE: {STEP_SIZE}")
    print("="*50)

def print_batch_config():
    """Print only batch size configuration for quick reference"""
    print("="*40)
    print("BATCH SIZE CONFIGURATION")
    print("="*40)
    print(f"REASONING_BATCH_SIZE: {REASONING_BATCH_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"REP_READING_BATCH_SIZE: {REP_READING_BATCH_SIZE}")
    print("="*40)

if __name__ == "__main__":
    print_config()
