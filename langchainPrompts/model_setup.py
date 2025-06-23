"""
Model Setup for Open Source LLMs
This module provides configurations for different open source models.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_community.llms import Ollama, HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load environment variables
load_dotenv()

class ModelManager:
    """Manager for different open source model configurations."""
    
    def __init__(self):
        self.models = {}
        self.embeddings = {}
    
    def get_ollama_model(self, model_name: str = "llama2", **kwargs) -> Ollama:
        """Get an Ollama model instance."""
        if model_name not in self.models:
            self.models[model_name] = Ollama(
                model=model_name,
                temperature=kwargs.get('temperature', 0.7),
                **kwargs
            )
        return self.models[model_name]
    
    def get_huggingface_model(self, model_name: str, **kwargs) -> HuggingFacePipeline:
        """Get a Hugging Face model instance."""
        cache_key = f"hf_{model_name}"
        if cache_key not in self.models:
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    **kwargs.get('model_kwargs', {})
                )
                
                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=kwargs.get('max_new_tokens', 512),
                    temperature=kwargs.get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                self.models[cache_key] = HuggingFacePipeline(
                    pipeline=pipe,
                    model_kwargs=kwargs.get('model_kwargs', {})
                )
                
            except Exception as e:
                print(f"Error loading Hugging Face model {model_name}: {e}")
                # Fallback to a smaller model
                return self.get_huggingface_model("microsoft/DialoGPT-medium", **kwargs)
        
        return self.models[cache_key]
    
    def get_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
        """Get embeddings model."""
        if model_name not in self.embeddings:
            self.embeddings[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
        return self.embeddings[model_name]
    
    def get_ollama_embeddings(self, model_name: str = "llama2") -> OllamaEmbeddings:
        """Get Ollama embeddings."""
        cache_key = f"ollama_emb_{model_name}"
        if cache_key not in self.embeddings:
            self.embeddings[cache_key] = OllamaEmbeddings(model=model_name)
        return self.embeddings[cache_key]

# Predefined model configurations
MODEL_CONFIGS = {
    "ollama": {
        "llama2": {
            "name": "llama2",
            "description": "Meta's Llama 2 model via Ollama",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "mistral": {
            "name": "mistral",
            "description": "Mistral AI's 7B model via Ollama",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "codellama": {
            "name": "codellama",
            "description": "Code Llama for programming tasks via Ollama",
            "temperature": 0.3,
            "max_tokens": 2048
        },
        "phi": {
            "name": "phi",
            "description": "Microsoft's Phi-2 model via Ollama",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    },
    "huggingface": {
        "microsoft/DialoGPT-medium": {
            "name": "microsoft/DialoGPT-medium",
            "description": "Microsoft's DialoGPT for conversational AI",
            "temperature": 0.7,
            "max_new_tokens": 512
        },
        "gpt2": {
            "name": "gpt2",
            "description": "OpenAI's GPT-2 model",
            "temperature": 0.7,
            "max_new_tokens": 512
        },
        "EleutherAI/gpt-neo-125M": {
            "name": "EleutherAI/gpt-neo-125M",
            "description": "GPT-Neo 125M parameter model",
            "temperature": 0.7,
            "max_new_tokens": 512
        },
        "microsoft/DialoGPT-small": {
            "name": "microsoft/DialoGPT-small",
            "description": "Smaller DialoGPT model for faster inference",
            "temperature": 0.7,
            "max_new_tokens": 256
        }
    }
}

# Global model manager instance
model_manager = ModelManager()

def get_model(model_type: str = "ollama", model_name: str = None, **kwargs):
    """Get a model instance based on type and name."""
    if model_type == "ollama":
        if model_name is None:
            model_name = "llama2"
        return model_manager.get_ollama_model(model_name, **kwargs)
    elif model_type == "huggingface":
        if model_name is None:
            model_name = "microsoft/DialoGPT-medium"
        return model_manager.get_huggingface_model(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_embeddings_model(model_type: str = "huggingface", model_name: str = None):
    """Get an embeddings model instance."""
    if model_type == "ollama":
        if model_name is None:
            model_name = "llama2"
        return model_manager.get_ollama_embeddings(model_name)
    elif model_type == "huggingface":
        if model_name is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return model_manager.get_embeddings(model_name)
    else:
        raise ValueError(f"Unsupported embeddings type: {model_type}")

def list_available_models():
    """List all available model configurations."""
    print("Available Models:")
    print("\nOllama Models:")
    for name, config in MODEL_CONFIGS["ollama"].items():
        print(f"  - {name}: {config['description']}")
    
    print("\nHugging Face Models:")
    for name, config in MODEL_CONFIGS["huggingface"].items():
        print(f"  - {name}: {config['description']}")

def check_model_availability():
    """Check which models are available on the system."""
    print("Checking model availability...")
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        print(f"\nOllama models available: {[m['name'] for m in models['models']]}")
    except ImportError:
        print("\nOllama not installed. Install with: pip install ollama")
    except Exception as e:
        print(f"\nOllama error: {e}")
    
    # Check Hugging Face
    try:
        from transformers import AutoTokenizer
        print("\nHugging Face models will be downloaded on first use.")
    except ImportError:
        print("\nHugging Face transformers not installed.")

if __name__ == "__main__":
    list_available_models()
    check_model_availability() 