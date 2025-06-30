"""
Example 02: Custom Runnable Classes and Functions
================================================

This example shows how to create custom runnable components.
You can make any function or class a runnable by implementing the Runnable interface
or using decorators. This allows you to integrate custom logic into LangChain chains.

Key Concepts:
- RunnableLambda: Convert any function into a runnable
- RunnablePassthrough: Pass through input without modification
- Custom Runnable classes: Create reusable components
- RunnableConfig: Configuration for runnable execution
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import re
import json

def text_analyzer(text: str) -> Dict[str, Any]:
    """
    Custom function that analyzes text and returns various metrics.
    This function will be converted into a runnable.
    """
    # Count words
    word_count = len(text.split())
    
    # Count sentences (simple approach)
    sentence_count = len(re.split(r'[.!?]+', text))
    
    # Count characters
    char_count = len(text)
    
    # Detect language (simple heuristic)
    language = "English"  # In a real app, you'd use a language detection library
    
    # Calculate average word length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "language": language,
        "avg_word_length": round(avg_word_length, 2)
    }

def sentiment_analyzer(text: str) -> str:
    """
    Simple sentiment analyzer function.
    In a real application, you'd use a proper sentiment analysis model.
    """
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "disappointing"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def format_analysis_result(input_dict: Dict[str, Any]) -> str:
    """
    Formats the analysis results into a readable string.
    """
    text = input_dict["text"]
    analysis = input_dict["analysis"]
    sentiment = input_dict["sentiment"]
    
    result = f"""
Text Analysis Results:
=====================
Text: "{text[:100]}{'...' if len(text) > 100 else ''}"

Statistics:
- Words: {analysis['word_count']}
- Sentences: {analysis['sentence_count']}
- Characters: {analysis['char_count']}
- Average word length: {analysis['avg_word_length']} characters
- Language: {analysis['language']}

Sentiment: {sentiment.capitalize()}
"""
    return result

def custom_runnable_example():
    """
    Demonstrates how to create and use custom runnable functions.
    """
    
    # Convert our custom functions into runnables
    text_analyzer_runnable = RunnableLambda(text_analyzer)
    sentiment_analyzer_runnable = RunnableLambda(sentiment_analyzer)
    formatter_runnable = RunnableLambda(format_analysis_result)
    
    # Create a chain that combines multiple custom runnables
    # This chain will:
    # 1. Take text input
    # 2. Analyze the text (word count, sentences, etc.)
    # 3. Analyze sentiment
    # 4. Format the results
    
    # We need to structure the data properly for the formatter
    def combine_results(input_data):
        return {
            "text": input_data["text"],
            "analysis": input_data["analysis"],
            "sentiment": input_data["sentiment"]
        }
    
    combine_runnable = RunnableLambda(combine_results)
    
    # Create the chain
    chain = (
        {
            "text": RunnablePassthrough(),
            "analysis": RunnablePassthrough() | text_analyzer_runnable,
            "sentiment": RunnablePassthrough() | sentiment_analyzer_runnable
        }
        | combine_runnable
        | formatter_runnable
    )
    
    # Test the chain
    test_text = "I love this amazing product! It's absolutely wonderful and makes me so happy. The quality is excellent and I would definitely recommend it to everyone."
    
    result = chain.invoke({"text": test_text})
    
    print("=== Custom Runnable Example ===")
    print(result)
    print("=" * 50)
    
    return result

class TextProcessor:
    """
    Custom Runnable class that demonstrates how to create reusable components.
    This class implements the Runnable interface by defining invoke and stream methods.
    """
    
    def __init__(self, operation: str):
        """
        Initialize the text processor with a specific operation.
        
        Args:
            operation: The type of processing to perform ('uppercase', 'lowercase', 'reverse')
        """
        self.operation = operation
    
    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process the input text according to the specified operation.
        This method is called when the runnable is invoked.
        """
        text = input_data.get("text", "")
        
        if self.operation == "uppercase":
            processed_text = text.upper()
        elif self.operation == "lowercase":
            processed_text = text.lower()
        elif self.operation == "reverse":
            processed_text = text[::-1]
        else:
            processed_text = text
        
        return {
            "original_text": text,
            "processed_text": processed_text,
            "operation": self.operation
        }
    
    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Stream the processing results.
        This method is called when the runnable is streamed.
        """
        result = self.invoke(input_data, config)
        yield result

def custom_class_example():
    """
    Demonstrates how to use custom Runnable classes.
    """
    
    # Create instances of our custom text processor
    uppercase_processor = TextProcessor("uppercase")
    reverse_processor = TextProcessor("reverse")
    
    # Create a chain that processes text in multiple ways
    chain = (
        {
            "uppercase_result": RunnablePassthrough() | uppercase_processor,
            "reverse_result": RunnablePassthrough() | reverse_processor
        }
    )
    
    # Test the chain
    test_text = {"text": "Hello, World!"}
    result = chain.invoke(test_text)
    
    print("=== Custom Runnable Class Example ===")
    print(f"Original text: {test_text['text']}")
    print(f"Uppercase: {result['uppercase_result']['processed_text']}")
    print(f"Reverse: {result['reverse_result']['processed_text']}")
    print("=" * 50)
    
    return result

if __name__ == "__main__":
    # Run the custom function example
    custom_runnable_example()
    
    # Run the custom class example
    custom_class_example() 