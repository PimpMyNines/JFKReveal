#!/usr/bin/env python3
"""
Setup script for downloading and configuring all models needed for JFKReveal.
This script will download the required Ollama models and set up the model configuration.
"""

import os
import sys
import argparse
import subprocess
import logging
from enum import Enum

# Add parent directory to path to allow importing the model configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.jfkreveal.utils.model_registry import ModelRegistry, ModelProvider, ModelType, ModelInfo
from src.jfkreveal.utils.model_config import ModelConfiguration, ReportType, AnalysisTask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup-models')

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Model sets for different use cases
MODEL_SETS = {
    "minimal": {
        "embedding": ("all-minilm", ModelProvider.OLLAMA),
        "chat": ("llama3", ModelProvider.OLLAMA),
        "multi_model": False,
        "models_to_compare": [],
    },
    "standard": {
        "embedding": ("nomic-embed-text", ModelProvider.OLLAMA),
        "chat": ("llama3", ModelProvider.OLLAMA),
        "multi_model": False,
        "models_to_compare": [],
    },
    "premium": {
        "embedding": ("text-embedding-3-small", ModelProvider.OPENAI),
        "chat": ("gpt-4o-mini", ModelProvider.OPENAI),
        "multi_model": False,
        "models_to_compare": [],
    },
    "multi": {
        "embedding": ("nomic-embed-text", ModelProvider.OLLAMA),
        "chat": ("llama3", ModelProvider.OLLAMA),
        "multi_model": True,
        "models_to_compare": ["gpt-4o", "llama3", "mistral"],
        "consolidated_model": "gpt-4o",
    },
    "local-multi": {
        "embedding": ("nomic-embed-text", ModelProvider.OLLAMA),
        "chat": ("llama3", ModelProvider.OLLAMA),
        "multi_model": True,
        "models_to_compare": ["llama3", "mistral", "gemma"],
        "consolidated_model": "llama3",
    },
}

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except Exception:
        return False

def download_ollama_models(models):
    """Download and verify Ollama models."""
    logger.info(f"Downloading {len(models)} Ollama models...")
    
    successes = 0
    failures = 0
    
    for model in models:
        try:
            logger.info(f"{Colors.BLUE}Downloading {model}...{Colors.ENDC}")
            result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logger.info(f"{Colors.GREEN}Successfully downloaded {model}{Colors.ENDC}")
                successes += 1
            else:
                logger.error(f"{Colors.RED}Failed to download {model}: {result.stderr}{Colors.ENDC}")
                failures += 1
        except Exception as e:
            logger.error(f"{Colors.RED}Error downloading {model}: {e}{Colors.ENDC}")
            failures += 1
    
    return successes, failures

def setup_config_for_set(config, set_name):
    """Set up model configuration based on the selected model set."""
    if set_name not in MODEL_SETS:
        logger.error(f"{Colors.RED}Unknown model set: {set_name}{Colors.ENDC}")
        return False
    
    model_set = MODEL_SETS[set_name]
    
    # Set prefer_local based on model set
    prefer_local = set_name not in ["premium"]
    config.set_prefer_local(prefer_local)
    logger.info(f"Prefer local models: {prefer_local}")
    
    # Set embedding model
    embed_model, embed_provider = model_set["embedding"]
    config.set_embedding_model(embed_model, embed_provider)
    logger.info(f"Embedding model set to {embed_model} ({embed_provider.value})")
    
    # Set task models
    chat_model, chat_provider = model_set["chat"]
    for task in AnalysisTask:
        config.set_model_for_task(task, chat_model, chat_provider)
    logger.info(f"All task models set to {chat_model} ({chat_provider.value})")
    
    # Set report configuration
    if model_set["multi_model"]:
        # Set report type to multi-model comparison
        config.set_report_type(ReportType.MULTI_MODEL_COMPARISON)
        # Enable multi-model reports
        config.enable_multi_model_reports(True, model_set["models_to_compare"])
        logger.info(f"Multi-model reports enabled with models: {', '.join(model_set['models_to_compare'])}")
        
        # Set consolidated model if specified
        if "consolidated_model" in model_set:
            if "report_configuration" not in config.config:
                config.config["report_configuration"] = {}
            config.config["report_configuration"]["consolidated_model"] = model_set["consolidated_model"]
            config.save_config()
            logger.info(f"Consolidated model set to {model_set['consolidated_model']}")
    else:
        # Set report type to standard
        config.set_report_type(ReportType.STANDARD)
        # Disable multi-model reports
        config.enable_multi_model_reports(False)
        logger.info("Standard report type configured")
    
    return True

def get_required_ollama_models(set_name):
    """Get the list of Ollama models required for the selected model set."""
    if set_name not in MODEL_SETS:
        return []
    
    model_set = MODEL_SETS[set_name]
    models = []
    
    # Add embedding model if it's from Ollama
    embed_model, embed_provider = model_set["embedding"]
    if embed_provider == ModelProvider.OLLAMA:
        models.append(embed_model)
    
    # Add chat model if it's from Ollama
    chat_model, chat_provider = model_set["chat"]
    if chat_provider == ModelProvider.OLLAMA:
        models.append(chat_model)
    
    # Add models to compare if multi-model is enabled
    if model_set["multi_model"]:
        for model in model_set["models_to_compare"]:
            # Skip OpenAI models
            if not model.startswith("gpt-"):
                if model not in models:
                    models.append(model)
    
    return models

def main():
    parser = argparse.ArgumentParser(description="Setup models for JFKReveal")
    parser.add_argument("--model-set", choices=list(MODEL_SETS.keys()), default="standard",
                        help="Model set to use (default: standard)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download models without changing configuration")
    parser.add_argument("--config-only", action="store_true",
                        help="Only update configuration without downloading models")
    parser.add_argument("--list-models", action="store_true",
                        help="List available model sets")
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{Colors.HEADER}{Colors.BOLD}JFKReveal Model Setup{Colors.ENDC}\n")
    
    # List model sets if requested
    if args.list_models:
        print(f"{Colors.CYAN}Available model sets:{Colors.ENDC}\n")
        for set_name, model_set in MODEL_SETS.items():
            embed_model, embed_provider = model_set["embedding"]
            chat_model, chat_provider = model_set["chat"]
            multi_model = model_set["multi_model"]
            
            # Format details
            if multi_model:
                models_to_compare = ", ".join(model_set["models_to_compare"])
                multi_desc = f" with models: {models_to_compare}"
            else:
                multi_desc = ""
            
            print(f"{Colors.BOLD}{set_name}{Colors.ENDC}:")
            print(f"  Embedding: {embed_model} ({embed_provider.value})")
            print(f"  Chat: {chat_model} ({chat_provider.value})")
            print(f"  Multi-model: {multi_model}{multi_desc}")
            print()
        
        sys.exit(0)
    
    # Check if Ollama is installed and running
    if not args.config_only:
        if not check_ollama_installed():
            logger.error(f"{Colors.RED}Ollama is not installed. Please install Ollama first: https://ollama.com/download{Colors.ENDC}")
            sys.exit(1)
        
        if not check_ollama_running():
            logger.error(f"{Colors.RED}Ollama is not running. Please start Ollama first.{Colors.ENDC}")
            logger.info(f"{Colors.YELLOW}On macOS: Run the Ollama app{Colors.ENDC}")
            logger.info(f"{Colors.YELLOW}On Linux: Run 'ollama serve' in a terminal{Colors.ENDC}")
            sys.exit(1)
    
    # Get required Ollama models for the selected set
    required_ollama_models = get_required_ollama_models(args.model_set)
    
    # Download models if requested
    if not args.config_only and required_ollama_models:
        logger.info(f"Downloading models for {args.model_set} model set...")
        successes, failures = download_ollama_models(required_ollama_models)
        
        if failures > 0:
            logger.warning(f"{Colors.YELLOW}Some models failed to download ({failures} failures, {successes} successes).{Colors.ENDC}")
        else:
            logger.info(f"{Colors.GREEN}All models downloaded successfully!{Colors.ENDC}")
    
    # Update configuration if requested
    if not args.download_only:
        logger.info(f"Setting up configuration for {args.model_set} model set...")
        config = ModelConfiguration()
        if setup_config_for_set(config, args.model_set):
            logger.info(f"{Colors.GREEN}Configuration updated successfully!{Colors.ENDC}")
    
    # Print success message
    if not args.download_only and not args.config_only:
        logger.info(f"""
{Colors.GREEN}✅ Setup complete!{Colors.ENDC}

You can now use JFKReveal with the {args.model_set} model set.

To view the current configuration:
    python tools/model_config_cli.py show
    
To run JFKReveal with this configuration:
    make run
    
For more information about models:
    python tools/model_config_cli.py list
""")

if __name__ == "__main__":
    main()