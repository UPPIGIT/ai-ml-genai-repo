"""
Basic LangChain Prompt Examples with Open Source Models
This file demonstrates fundamental prompt concepts using open source LLMs.
"""

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from model_setup import get_model, list_available_models

# Load environment variables
load_dotenv()

def simple_prompt_example(model_type="ollama", model_name="llama2"):
    """Demonstrate a simple text prompt with open source models."""
    print(f"=== Simple Prompt Example ({model_type}: {model_name}) ===")
    
    # Get the model
    llm = get_model(model_type, model_name)
    
    prompt = "Explain quantum computing in simple terms."
    print(f"Prompt: {prompt}")
    
    try:
        response = llm.invoke(prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        print("Trying with a different model...")
        # Fallback to a smaller model
        try:
            llm = get_model("huggingface", "microsoft/DialoGPT-small")
            response = llm.invoke(prompt)
            print(f"Response (fallback): {response}\n")
        except Exception as e2:
            print(f"Fallback also failed: {e2}\n")

def template_prompt_example(model_type="ollama", model_name="llama2"):
    """Demonstrate prompt templates with variables."""
    print(f"=== Template Prompt Example ({model_type}: {model_name}) ===")
    
    # Get the model
    llm = get_model(model_type, model_name)
    
    # Create a prompt template
    template = """
    You are a helpful assistant that explains technical concepts.
    
    Please explain {concept} in a way that a {audience} can understand.
    Keep your explanation under {word_limit} words.
    """
    
    prompt_template = PromptTemplate(
        input_variables=["concept", "audience", "word_limit"],
        template=template
    )
    
    # Format the prompt with variables
    formatted_prompt = prompt_template.format(
        concept="machine learning",
        audience="high school student",
        word_limit="100"
    )
    
    print(f"Formatted Prompt:\n{formatted_prompt}")
    
    try:
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

def chat_prompt_template_example(model_type="ollama", model_name="llama2"):
    """Demonstrate chat prompt templates."""
    print(f"=== Chat Prompt Template Example ({model_type}: {model_name}) ===")
    
    # Get the model
    llm = get_model(model_type, model_name)
    
    # Create a chat prompt template
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a coding tutor who helps students understand programming concepts."),
        ("human", "Explain {concept} with a practical example in {language}.")
    ])
    
    # Format the chat prompt
    messages = chat_template.format_messages(
        concept="recursion",
        language="Python"
    )
    
    print("Chat Messages:")
    for message in messages:
        print(f"{message.type}: {message.content}")
    
    try:
        response = llm.invoke(messages)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

def few_shot_prompt_example(model_type="ollama", model_name="llama2"):
    """Demonstrate few-shot learning with examples."""
    print(f"=== Few-Shot Prompt Example ({model_type}: {model_name}) ===")
    
    # Get the model
    llm = get_model(model_type, model_name)
    
    # Define examples for sentiment analysis
    examples = [
        {"text": "I love this product!", "sentiment": "positive"},
        {"text": "This is terrible quality.", "sentiment": "negative"},
        {"text": "It's okay, nothing special.", "sentiment": "neutral"},
        {"text": "Amazing experience, highly recommend!", "sentiment": "positive"},
        {"text": "Disappointed with the service.", "sentiment": "negative"}
    ]
    
    # Create the example prompt template
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="Text: {text}\nSentiment: {sentiment}"
    )
    
    # Create the few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Analyze the sentiment of the following text. Classify it as positive, negative, or neutral.",
        suffix="Text: {input_text}\nSentiment:",
        input_variables=["input_text"],
        example_separator="\n\n"
    )
    
    # Test the few-shot prompt
    test_text = "This movie exceeded all my expectations!"
    formatted_prompt = few_shot_prompt.format(input_text=test_text)
    
    print(f"Few-Shot Prompt:\n{formatted_prompt}")
    
    try:
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")

def dynamic_few_shot_example(model_type="ollama", model_name="llama2"):
    """Demonstrate dynamic few-shot example selection."""
    print(f"=== Dynamic Few-Shot Example ({model_type}: {model_name}) ===")
    
    # Get the model
    llm = get_model(model_type, model_name)
    
    # More examples for dynamic selection
    examples = [
        {"input": "2 + 2", "output": "4"},
        {"input": "5 * 3", "output": "15"},
        {"input": "10 / 2", "output": "5"},
        {"input": "7 - 4", "output": "3"},
        {"input": "3^2", "output": "9"},
        {"input": "sqrt(16)", "output": "4"},
        {"input": "15 % 4", "output": "3"},
        {"input": "2^3", "output": "8"},
        {"input": "20 / 5", "output": "4"},
        {"input": "6 + 9", "output": "15"}
    ]
    
    # Create example selector based on length
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        max_length=200,
        get_text_length=lambda x: len(str(x))
    )
    
    # Create the example prompt template
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    )
    
    # Create dynamic few-shot prompt
    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Solve the following math problems:",
        suffix="Input: {input}\nOutput:",
        input_variables=["input"],
        example_separator="\n\n"
    )
    
    # Test with different inputs
    test_inputs = ["8 + 12", "25 / 5", "3^3"]
    
    for test_input in test_inputs:
        formatted_prompt = dynamic_prompt.format(input=test_input)
        print(f"Prompt for '{test_input}':\n{formatted_prompt}")
        
        try:
            response = llm.invoke(formatted_prompt)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

def model_comparison_example():
    """Compare different open source models."""
    print("=== Model Comparison Example ===")
    
    prompt = "Write a short poem about artificial intelligence."
    
    models_to_test = [
        ("ollama", "llama2"),
        ("ollama", "mistral"),
        ("huggingface", "microsoft/DialoGPT-medium"),
        ("huggingface", "gpt2")
    ]
    
    for model_type, model_name in models_to_test:
        print(f"\n--- Testing {model_type}: {model_name} ---")
        try:
            llm = get_model(model_type, model_name)
            response = llm.invoke(prompt)
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run all basic prompt examples with open source models."""
    print("LangChain Basic Prompt Examples with Open Source Models\n")
    print("=" * 60)
    
    # List available models
    list_available_models()
    
    try:
        # Test with Ollama (if available)
        print("\n" + "="*60)
        print("Testing with Ollama models...")
        simple_prompt_example("ollama", "llama2")
        template_prompt_example("ollama", "llama2")
        chat_prompt_template_example("ollama", "llama2")
        few_shot_prompt_example("ollama", "llama2")
        dynamic_few_shot_example("ollama", "llama2")
        
    except Exception as e:
        print(f"Ollama test failed: {e}")
    
    try:
        # Test with Hugging Face models
        print("\n" + "="*60)
        print("Testing with Hugging Face models...")
        simple_prompt_example("huggingface", "microsoft/DialoGPT-medium")
        template_prompt_example("huggingface", "microsoft/DialoGPT-medium")
        chat_prompt_template_example("huggingface", "microsoft/DialoGPT-medium")
        few_shot_prompt_example("huggingface", "microsoft/DialoGPT-medium")
        dynamic_few_shot_example("huggingface", "microsoft/DialoGPT-medium")
        
    except Exception as e:
        print(f"Hugging Face test failed: {e}")
    
    # Model comparison
    try:
        model_comparison_example()
    except Exception as e:
        print(f"Model comparison failed: {e}")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main() 