#!/usr/bin/env python3
"""
Setup script for installing and configuring Ollama with the recommended embedding model.
"""

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup-ollama')

# Best embedding models to use with Ollama
RECOMMENDED_MODELS = {
    "nomic-embed-text": {
        "description": "High-performing open embedding model with 8K context window (137M parameters)",
        "size": "274MB",
        "command": "ollama pull nomic-embed-text",
    },
    "mxbai-embed-large": {
        "description": "State-of-the-art large embedding model (334M parameters)",
        "size": "680MB",
        "command": "ollama pull mxbai-embed-large",
    },
    "all-minilm": {
        "description": "Lightweight embedding model, good for low-resource environments (22M parameters)",
        "size": "50MB",
        "command": "ollama pull all-minilm",
    }
}

DEFAULT_MODEL = "nomic-embed-text"

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

def download_model(model_name):
    """Download the specified model from Ollama."""
    model_info = RECOMMENDED_MODELS.get(model_name)
    if not model_info:
        logger.error(f"Unknown model: {model_name}")
        return False
    
    logger.info(f"Downloading {model_name} ({model_info['size']})...")
    logger.info(f"Description: {model_info['description']}")
    
    try:
        subprocess.run(model_info["command"], shell=True, check=True)
        logger.info(f"Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False

def update_env_file(model_name):
    """Update .env file with the selected model."""
    env_file = ".env"
    env_example = ".env.example"
    
    # Create .env from example if it doesn't exist
    if not os.path.exists(env_file) and os.path.exists(env_example):
        with open(env_example, "r") as example, open(env_file, "w") as env:
            env.write(example.read())
    
    # Read current .env
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            lines = f.readlines()
        
        # Update or add Ollama config
        found_provider = False
        found_model = False
        
        for i, line in enumerate(lines):
            if line.startswith("EMBEDDING_PROVIDER="):
                lines[i] = f"EMBEDDING_PROVIDER=ollama\n"
                found_provider = True
            elif line.startswith("OLLAMA_EMBEDDING_MODEL="):
                lines[i] = f"OLLAMA_EMBEDDING_MODEL={model_name}\n"
                found_model = True
        
        # Add lines if not found
        if not found_provider:
            lines.append(f"EMBEDDING_PROVIDER=ollama\n")
        if not found_model:
            lines.append(f"OLLAMA_EMBEDDING_MODEL={model_name}\n")
        
        # Write updated .env file
        with open(env_file, "w") as f:
            f.writelines(lines)
            
        logger.info(f"Updated {env_file} with Ollama embedding model: {model_name}")
    else:
        logger.warning(f"No {env_file} file found. Please create one manually.")

def main():
    parser = argparse.ArgumentParser(description="Setup Ollama with recommended embedding model")
    parser.add_argument("--model", choices=list(RECOMMENDED_MODELS.keys()), default=DEFAULT_MODEL,
                        help="Embedding model to install (default: %(default)s)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, info in RECOMMENDED_MODELS.items():
            print(f"  {name}: {info['description']} ({info['size']})")
        return 0
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        logger.error("Ollama is not installed. Please install Ollama first: https://ollama.com/download")
        return 1
    
    # Check if Ollama is running
    if not check_ollama_running():
        logger.error("Ollama is not running. Please start Ollama first.")
        logger.info("On macOS: Run the Ollama app")
        logger.info("On Linux: Run 'ollama serve' in a terminal")
        return 1
    
    # Download the model
    if not download_model(args.model):
        return 1
    
    # Update .env file
    update_env_file(args.model)
    
    logger.info(f"""
✅ Setup complete!

You can now use JFKReveal with local embedding model: {args.model}

To run with this configuration:
    make run
    
For more information about Ollama embedding models:
    https://ollama.com/search?c=embedding
""")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 