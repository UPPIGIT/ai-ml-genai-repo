"""
Basic LangChain Prompt Examples
This file demonstrates fundamental prompt concepts in LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def simple_prompt_example():
    """Demonstrate a simple text prompt."""
    print("=== Simple Prompt Example ===")
    
    prompt = "Explain quantum computing in simple terms."
    response = llm.invoke(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.content}\n")

def template_prompt_example():
    """Demonstrate prompt templates with variables."""
    print("=== Template Prompt Example ===")
    
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
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}\n")

def chat_prompt_template_example():
    """Demonstrate chat prompt templates."""
    print("=== Chat Prompt Template Example ===")
    
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
    
    response = llm.invoke(messages)
    print(f"Response: {response.content}\n")

def few_shot_prompt_example():
    """Demonstrate few-shot learning with examples."""
    print("=== Few-Shot Prompt Example ===")
    
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
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}\n")

def dynamic_few_shot_example():
    """Demonstrate dynamic few-shot example selection."""
    print("=== Dynamic Few-Shot Example ===")
    
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
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}\n")

def main():
    """Run all basic prompt examples."""
    print("LangChain Basic Prompt Examples\n")
    print("=" * 50)
    
    try:
        simple_prompt_example()
        template_prompt_example()
        chat_prompt_template_example()
        few_shot_prompt_example()
        dynamic_few_shot_example()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 