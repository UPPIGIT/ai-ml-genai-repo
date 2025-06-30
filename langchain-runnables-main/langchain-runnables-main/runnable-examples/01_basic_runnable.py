"""
Example 01: Basic Runnable Chain
================================

This example demonstrates the fundamental concept of LangChain runnables.
A runnable is any object that can be invoked with an input and returns an output.
This includes models, chains, tools, and other components.

Key Concepts:
- Runnable interface: The base interface for all LangChain components
- Chain: A sequence of runnables that can be executed together
- Invoke: The method to execute a runnable with input
- Stream: The method to execute a runnable and get streaming output
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up your OpenAI API key (you'll need to set this in your environment)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def basic_runnable_example():
    """
    Demonstrates a basic runnable chain that:
    1. Takes a topic as input
    2. Generates a creative story about that topic
    3. Returns the story as output
    """
    
    # Step 1: Create a language model
    # This is a runnable that takes text input and returns AI-generated text
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7  # Higher temperature for more creative output
    )
    
    # Step 2: Create a prompt template
    # This is a runnable that takes variables and returns a formatted prompt
    prompt = ChatPromptTemplate.from_template(
        "Write a short, creative story about {topic}. "
        "Make it engaging and suitable for children. "
        "Keep it under 100 words."
    )
    
    # Step 3: Create an output parser
    # This is a runnable that takes the model output and formats it
    output_parser = StrOutputParser()
    
    # Step 4: Chain them together
    # The | operator creates a chain of runnables
    chain = prompt | model | output_parser
    
    # Step 5: Execute the chain
    # This is where the magic happens - all runnables execute in sequence
    result = chain.invoke({"topic": "a magical robot"})
    
    print("=== Basic Runnable Example ===")
    print(f"Input topic: a magical robot")
    print(f"Generated story:\n{result}")
    print("=" * 50)
    
    return result

def streaming_example():
    """
    Demonstrates streaming output from a runnable chain.
    Streaming allows you to see the output as it's being generated,
    which is useful for real-time applications.
    """
    
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5
    )
    
    prompt = ChatPromptTemplate.from_template(
        "Explain the concept of {concept} in simple terms. "
        "Use analogies and examples to make it clear."
    )
    
    chain = prompt | model | StrOutputParser()
    
    print("=== Streaming Example ===")
    print("Streaming explanation of 'artificial intelligence':")
    
    # Use stream() instead of invoke() for streaming output
    for chunk in chain.stream({"concept": "artificial intelligence"}):
        print(chunk, end="", flush=True)
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    # Run the basic example
    basic_runnable_example()
    
    # Run the streaming example
    streaming_example() 