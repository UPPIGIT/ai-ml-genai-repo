"""
Run All LangChain Prompt Examples
This script runs all the prompt examples in sequence.
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
                              capture_output=True, text=True, timeout=300)
        
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

def main():
    """Run all LangChain prompt examples."""
    print("LangChain Prompt Examples - Complete Suite")
    print("=" * 60)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("\nYou can copy from env_example.txt as a starting point.")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # List of examples to run
    examples = [
        ("basic_prompts.py", "Basic Prompt Examples"),
        ("advanced_prompts.py", "Advanced Prompt Examples"),
        ("prompt_templates.py", "Prompt Template Examples"),
        ("few_shot_examples.py", "Few-Shot Learning Examples"),
        ("chain_prompts.py", "Chain Prompt Examples"),
        ("output_parsing.py", "Output Parsing Examples"),
        ("prompt_optimization.py", "Prompt Optimization Examples")
    ]
    
    # Run each example
    for script_name, description in examples:
        if os.path.exists(script_name):
            run_example(script_name, description)
        else:
            print(f"❌ Script not found: {script_name}")
    
    print(f"\n{'='*60}")
    print("All examples completed!")
    print("=" * 60)
    
    print("\n📚 Summary of what you've learned:")
    print("• Basic prompt creation and templates")
    print("• Advanced techniques like chain of thought")
    print("• Few-shot learning with examples")
    print("• Multi-step reasoning chains")
    print("• Structured output parsing")
    print("• Prompt optimization and A/B testing")
    
    print("\n🚀 Next steps:")
    print("• Experiment with your own prompts")
    print("• Try different models and parameters")
    print("• Build your own LangChain applications")
    print("• Explore the LangChain documentation")

if __name__ == "__main__":
    main() 