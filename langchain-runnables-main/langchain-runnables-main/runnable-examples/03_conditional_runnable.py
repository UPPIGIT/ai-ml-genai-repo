"""
Example 03: Conditional Logic and Branching
==========================================

This example demonstrates how to implement conditional logic and branching
in LangChain runnable chains. You can create different execution paths
based on input conditions or intermediate results.

Key Concepts:
- RunnableBranch: Execute different runnables based on conditions
- RunnableLambda: Create conditional logic functions
- RunnablePassthrough: Pass data through the chain
- Conditional chains: Different execution paths based on input
"""

from typing import Dict, Any, Union
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json

def classify_input(input_data: Dict[str, Any]) -> str:
    """
    Classify the input based on its content and length.
    This function determines which processing path to take.
    """
    text = input_data.get("text", "")
    
    # Simple classification logic
    if len(text) < 50:
        return "short"
    elif len(text) < 200:
        return "medium"
    else:
        return "long"

def short_text_processor(input_data: Dict[str, Any]) -> str:
    """
    Process short text with a concise response.
    """
    text = input_data["text"]
    return f"Short text summary: '{text}' - This is a brief input that gets a concise response."

def medium_text_processor(input_data: Dict[str, Any]) -> str:
    """
    Process medium text with a detailed analysis.
    """
    text = input_data["text"]
    word_count = len(text.split())
    char_count = len(text)
    
    return f"""Medium text analysis:
Text: "{text}"
Word count: {word_count}
Character count: {char_count}
This text gets a moderate level of analysis."""

def long_text_processor(input_data: Dict[str, Any]) -> str:
    """
    Process long text with comprehensive analysis.
    """
    text = input_data["text"]
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    return f"""Comprehensive text analysis:
Text: "{text[:100]}..."
Word count: {word_count}
Character count: {char_count}
Sentence count: {sentence_count}
Average word length: {avg_word_length:.2f} characters
This long text receives detailed analysis."""

def conditional_processing_example():
    """
    Demonstrates conditional processing based on input classification.
    """
    
    # Create the classification function
    classifier = RunnableLambda(classify_input)
    
    # Create the processing functions
    short_processor = RunnableLambda(short_text_processor)
    medium_processor = RunnableLambda(medium_text_processor)
    long_processor = RunnableLambda(long_text_processor)
    
    # Create a branch that routes to different processors based on classification
    branch = RunnableBranch(
        (lambda x: x == "short", short_processor),
        (lambda x: x == "medium", medium_processor),
        (lambda x: x == "long", long_processor)
    )
    
    # Create the full chain
    chain = (
        {
            "text": RunnablePassthrough(),
            "classification": RunnablePassthrough() | classifier
        }
        | branch
    )
    
    # Test with different text lengths
    test_cases = [
        "Hello world!",  # Short
        "This is a medium length text that contains several words and should trigger the medium processing path.",  # Medium
        "This is a very long text that contains many words and sentences. It goes on and on with lots of content. The purpose is to demonstrate how the conditional processing works with different input lengths. This text should definitely trigger the long text processing path because it has so much content to analyze."  # Long
    ]
    
    print("=== Conditional Processing Example ===")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_text}")
        result = chain.invoke({"text": test_text})
        print(f"Result: {result}")
        print("-" * 40)
    
    print("=" * 50)

def sentiment_based_processing():
    """
    Demonstrates conditional processing based on sentiment analysis.
    """
    
    def analyze_sentiment(input_data: Dict[str, Any]) -> str:
        """Simple sentiment analysis"""
        text = input_data["text"].lower()
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love"]
        negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "disappointing"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def positive_response(input_data: Dict[str, Any]) -> str:
        """Generate a positive, encouraging response"""
        return f"Great to hear positive feedback! Your message: '{input_data['text']}' - Keep up the positive energy!"
    
    def negative_response(input_data: Dict[str, Any]) -> str:
        """Generate a supportive response for negative feedback"""
        return f"I understand your concerns about: '{input_data['text']}' - Let's work together to address these issues."
    
    def neutral_response(input_data: Dict[str, Any]) -> str:
        """Generate a neutral, informative response"""
        return f"Thank you for your feedback: '{input_data['text']}' - We appreciate your input and will consider it carefully."
    
    # Create the chain
    sentiment_analyzer = RunnableLambda(analyze_sentiment)
    
    sentiment_branch = RunnableBranch(
        (lambda x: x == "positive", RunnableLambda(positive_response)),
        (lambda x: x == "negative", RunnableLambda(negative_response)),
        (lambda x: x == "neutral", RunnableLambda(neutral_response))
    )
    
    chain = (
        {
            "text": RunnablePassthrough(),
            "sentiment": RunnablePassthrough() | sentiment_analyzer
        }
        | sentiment_branch
    )
    
    # Test with different sentiments
    test_cases = [
        "I love this product! It's amazing and wonderful!",
        "This is terrible and disappointing. I hate it.",
        "The product works as expected. It's okay."
    ]
    
    print("=== Sentiment-Based Processing Example ===")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_text}")
        result = chain.invoke({"text": test_text})
        print(f"Result: {result}")
        print("-" * 40)
    
    print("=" * 50)

def complex_conditional_chain():
    """
    Demonstrates a more complex conditional chain with multiple decision points.
    """
    
    def determine_topic(input_data: Dict[str, Any]) -> str:
        """Determine the topic of the text"""
        text = input_data["text"].lower()
        
        if any(word in text for word in ["python", "programming", "code", "software"]):
            return "programming"
        elif any(word in text for word in ["weather", "temperature", "rain", "sunny"]):
            return "weather"
        elif any(word in text for word in ["food", "cooking", "recipe", "restaurant"]):
            return "food"
        else:
            return "general"
    
    def determine_complexity(input_data: Dict[str, Any]) -> str:
        """Determine the complexity level"""
        text = input_data["text"]
        word_count = len(text.split())
        
        if word_count < 10:
            return "simple"
        elif word_count < 30:
            return "moderate"
        else:
            return "complex"
    
    # Create topic-specific processors
    def programming_processor(input_data: Dict[str, Any]) -> str:
        return f"Programming topic detected: '{input_data['text']}' - I can help with coding questions!"
    
    def weather_processor(input_data: Dict[str, Any]) -> str:
        return f"Weather topic detected: '{input_data['text']}' - Let me check the forecast for you."
    
    def food_processor(input_data: Dict[str, Any]) -> str:
        return f"Food topic detected: '{input_data['text']}' - I love talking about cooking and recipes!"
    
    def general_processor(input_data: Dict[str, Any]) -> str:
        return f"General topic: '{input_data['text']}' - I'm here to help with any questions."
    
    # Create complexity-specific formatters
    def simple_formatter(input_data: Dict[str, Any]) -> str:
        return f"Simple response: {input_data['topic_response']}"
    
    def moderate_formatter(input_data: Dict[str, Any]) -> str:
        return f"Moderate response: {input_data['topic_response']} (with additional context)"
    
    def complex_formatter(input_data: Dict[str, Any]) -> str:
        return f"Complex response: {input_data['topic_response']} (with detailed analysis and recommendations)"
    
    # Create the chains
    topic_detector = RunnableLambda(determine_topic)
    complexity_detector = RunnableLambda(determine_complexity)
    
    topic_branch = RunnableBranch(
        (lambda x: x == "programming", RunnableLambda(programming_processor)),
        (lambda x: x == "weather", RunnableLambda(weather_processor)),
        (lambda x: x == "food", RunnableLambda(food_processor)),
        (lambda x: x == "general", RunnableLambda(general_processor))
    )
    
    complexity_branch = RunnableBranch(
        (lambda x: x == "simple", RunnableLambda(simple_formatter)),
        (lambda x: x == "moderate", RunnableLambda(moderate_formatter)),
        (lambda x: x == "complex", RunnableLambda(complex_formatter))
    )
    
    # Create the full chain
    chain = (
        {
            "text": RunnablePassthrough(),
            "topic": RunnablePassthrough() | topic_detector,
            "complexity": RunnablePassthrough() | complexity_detector
        }
        | {
            "text": RunnablePassthrough(),
            "topic_response": topic_branch,
            "complexity": RunnablePassthrough()
        }
        | complexity_branch
    )
    
    # Test cases
    test_cases = [
        "Python is great",  # Simple programming
        "I need help with a complex software architecture problem involving multiple microservices and distributed systems",  # Complex programming
        "It's sunny today",  # Simple weather
        "The weather forecast shows scattered thunderstorms with temperatures ranging from 65 to 75 degrees Fahrenheit throughout the week",  # Complex weather
        "I want pizza",  # Simple food
        "I'm looking for a recipe for authentic Italian lasagna with homemade pasta sheets and traditional béchamel sauce"  # Complex food
    ]
    
    print("=== Complex Conditional Chain Example ===")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_text}")
        result = chain.invoke({"text": test_text})
        print(f"Result: {result}")
        print("-" * 40)
    
    print("=" * 50)

if __name__ == "__main__":
    # Run all conditional examples
    conditional_processing_example()
    sentiment_based_processing()
    complex_conditional_chain() 