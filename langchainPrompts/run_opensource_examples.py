"""
Run All LangChain Open Source Model Examples
This script runs all the prompt examples with open source models.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_example(script_name: str, description: str):
    """Run a single example script."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Success!")
            print(result.stdout)
        else:
            print("❌ Error occurred:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - script took too long to run")
    except Exception as e:
        print(f"❌ Exception: {e}")

def check_model_availability():
    """Check which models are available."""
    print("Checking model availability...")
    
    try:
        # Check Ollama
        import ollama
        models = ollama.list()
        print(f"✅ Ollama models available: {[m['name'] for m in models['models']]}")
        return True
    except ImportError:
        print("⚠️  Ollama not installed. Install with: pip install ollama")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
    
    try:
        # Check Hugging Face
        from transformers import AutoTokenizer
        print("✅ Hugging Face transformers available")
        return True
    except ImportError:
        print("⚠️  Hugging Face transformers not installed")
    
    return False

def setup_instructions():
    """Show setup instructions."""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Install Ollama (Recommended):")
    print("   # On macOS")
    print("   brew install ollama")
    print("   # On Linux")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("   # On Windows: Download from https://ollama.ai/download")
    
    print("\n2. Start Ollama and download models:")
    print("   ollama serve")
    print("   ollama pull llama2")
    print("   ollama pull mistral")
    
    print("\n3. Install project dependencies:")
    print("   uv sync")
    print("   uv sync --extra ollama")
    print("   uv sync --extra huggingface")
    
    print("\n4. Set up environment:")
    print("   cp env_example.txt .env")
    print("   # Edit .env with your preferences")
    
    print("\n5. Run examples:")
    print("   uv run python run_opensource_examples.py")

def main():
    """Run all open source model examples."""
    print("LangChain Open Source Model Examples - Complete Suite")
    print("=" * 60)
    
    # Check if models are available
    models_available = check_model_availability()
    
    if not models_available:
        print("\n❌ No models available!")
        setup_instructions()
        return
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("Please create a .env file with your model preferences:")
        print("cp env_example.txt .env")
        print("\nYou can use the default settings for testing.")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # List of examples to run
    examples = [
        ("basic_prompts_opensource.py", "Basic Prompt Examples with Open Source Models"),
        ("advanced_prompts_opensource.py", "Advanced Prompt Examples with Open Source Models"),
        ("model_setup.py", "Model Setup and Configuration"),
    ]
    
    # Run each example
    for script_name, description in examples:
        if os.path.exists(script_name):
            run_example(script_name, description)
        else:
            print(f"❌ Script not found: {script_name}")
    
    print(f"\n{'='*60}")
    print("All open source examples completed!")
    print("=" * 60)
    
    print("\n📚 Summary of what you've learned:")
    print("• Using open source models with LangChain")
    print("• Ollama integration for local model inference")
    print("• Hugging Face model integration")
    print("• Prompt engineering with open source models")
    print("• Model comparison and optimization")
    
    print("\n🚀 Next steps:")
    print("• Experiment with different open source models")
    print("• Try different prompt techniques")
    print("• Optimize prompts for your specific use case")
    print("• Explore the Hugging Face model hub")
    print("• Set up model serving for production use")
    
    print("\n🔧 Troubleshooting:")
    print("• If models are slow, try smaller models like 'phi' or 'DialoGPT-small'")
    print("• For better performance, use GPU acceleration")
    print("• Check model availability with: python model_setup.py")
    print("• See setup_opensource_models.md for detailed instructions")

if __name__ == "__main__":
    main() 