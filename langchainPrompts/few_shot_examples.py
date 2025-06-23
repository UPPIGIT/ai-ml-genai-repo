"""
LangChain Few-Shot Learning Examples
This file demonstrates various few-shot learning techniques and patterns.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize the language model and embeddings
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

def basic_few_shot_example():
    """Demonstrate basic few-shot learning."""
    print("=== Basic Few-Shot Example ===")
    
    # Define examples for text classification
    examples = [
        {"text": "I love this movie!", "sentiment": "positive"},
        {"text": "This is the worst film ever.", "sentiment": "negative"},
        {"text": "It was okay, nothing special.", "sentiment": "neutral"},
        {"text": "Absolutely fantastic performance!", "sentiment": "positive"},
        {"text": "Terrible acting and poor plot.", "sentiment": "negative"}
    ]
    
    # Create example prompt template
    example_prompt = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="Text: {text}\nSentiment: {sentiment}"
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Analyze the sentiment of the following text. Classify it as positive, negative, or neutral.",
        suffix="Text: {input_text}\nSentiment:",
        input_variables=["input_text"],
        example_separator="\n\n"
    )
    
    # Test the few-shot prompt
    test_texts = [
        "This product exceeded all my expectations!",
        "I'm disappointed with the quality.",
        "It's a decent option for the price."
    ]
    
    for test_text in test_texts:
        formatted_prompt = few_shot_prompt.format(input_text=test_text)
        print(f"Input: {test_text}")
        print(f"Few-Shot Prompt:\n{formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}\n")

def length_based_example_selection():
    """Demonstrate length-based example selection."""
    print("=== Length-Based Example Selection ===")
    
    # Define many examples with varying complexity
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
        {"input": "6 + 9", "output": "15"},
        {"input": "log(100)", "output": "2"},
        {"input": "sin(90)", "output": "1"},
        {"input": "cos(0)", "output": "1"},
        {"input": "tan(45)", "output": "1"},
        {"input": "e^0", "output": "1"}
    ]
    
    # Create example selector
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        max_length=150,
        get_text_length=lambda x: len(str(x))
    )
    
    # Create example prompt template
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Solve the following math problems:",
        suffix="Input: {input}\nOutput:",
        input_variables=["input"],
        example_separator="\n\n"
    )
    
    # Test with different inputs
    test_inputs = ["8 + 12", "25 / 5", "3^3", "log(1000)"]
    
    for test_input in test_inputs:
        formatted_prompt = few_shot_prompt.format(input=test_input)
        print(f"Input: {test_input}")
        print(f"Selected Examples: {len(example_selector.select_examples({'input': test_input}))}")
        print(f"Prompt Length: {len(formatted_prompt)} characters")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}\n")

def semantic_similarity_example_selection():
    """Demonstrate semantic similarity-based example selection."""
    print("=== Semantic Similarity Example Selection ===")
    
    # Define examples for code generation
    examples = [
        {"task": "Create a function to calculate factorial", "language": "Python", "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"},
        {"task": "Write a function to find the maximum number in a list", "language": "Python", "code": "def find_max(numbers):\n    return max(numbers) if numbers else None"},
        {"task": "Create a function to reverse a string", "language": "Python", "code": "def reverse_string(s):\n    return s[::-1]"},
        {"task": "Write a function to check if a number is prime", "language": "Python", "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"},
        {"task": "Create a function to calculate fibonacci numbers", "language": "Python", "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
        {"task": "Write a function to sort a list", "language": "Python", "code": "def sort_list(lst):\n    return sorted(lst)"},
        {"task": "Create a function to count vowels in a string", "language": "Python", "code": "def count_vowels(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)"},
        {"task": "Write a function to find the GCD of two numbers", "language": "Python", "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"}
    ]
    
    # Create example selector using semantic similarity
    example_selector = SemanticSimilarityExampleSelector(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=FAISS,
        k=3
    )
    
    # Create example prompt template
    example_prompt = PromptTemplate(
        input_variables=["task", "language", "code"],
        template="Task: {task}\nLanguage: {language}\nCode:\n{code}"
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Generate code for the following task:",
        suffix="Task: {task}\nLanguage: {language}\nCode:",
        input_variables=["task", "language"],
        example_separator="\n\n"
    )
    
    # Test with different tasks
    test_tasks = [
        ("Create a function to calculate the sum of digits in a number", "Python"),
        ("Write a function to check if a string is a palindrome", "Python"),
        ("Create a function to find the LCM of two numbers", "Python")
    ]
    
    for task, language in test_tasks:
        formatted_prompt = few_shot_prompt.format(task=task, language=language)
        print(f"Task: {task}")
        print(f"Language: {language}")
        print(f"Selected Examples: {len(example_selector.select_examples({'task': task}))}")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}\n")

def pattern_matching_examples():
    """Demonstrate pattern matching with few-shot examples."""
    print("=== Pattern Matching Examples ===")
    
    # Define patterns for different types of responses
    patterns = {
        "problem_solution": [
            {
                "problem": "User can't log in to the application",
                "analysis": "This is likely a authentication or session management issue",
                "steps": "1. Check credentials\n2. Verify session tokens\n3. Clear browser cache",
                "solution": "Reset password and clear browser cache"
            },
            {
                "problem": "Database connection is slow",
                "analysis": "This could be due to network issues or database optimization",
                "steps": "1. Check network connectivity\n2. Analyze query performance\n3. Review database indexes",
                "solution": "Optimize database queries and add proper indexing"
            }
        ],
        "code_review": [
            {
                "code": "for i in range(len(items)):\n    print(items[i])",
                "issues": "Inefficient iteration, not using enumerate",
                "suggestions": "Use enumerate for better readability and performance",
                "improved": "for i, item in enumerate(items):\n    print(item)"
            },
            {
                "code": "if x > 0:\n    return True\nelse:\n    return False",
                "issues": "Unnecessary if-else statement",
                "suggestions": "Direct boolean return",
                "improved": "return x > 0"
            }
        ]
    }
    
    # Create pattern-specific templates
    for pattern_name, pattern_examples in patterns.items():
        print(f"\n--- {pattern_name.upper()} PATTERN ---")
        
        if pattern_name == "problem_solution":
            example_prompt = PromptTemplate(
                input_variables=["problem", "analysis", "steps", "solution"],
                template="Problem: {problem}\nAnalysis: {analysis}\nSteps: {steps}\nSolution: {solution}"
            )
            
            few_shot_prompt = FewShotPromptTemplate(
                examples=pattern_examples,
                example_prompt=example_prompt,
                prefix="Analyze the following problem and provide a structured solution:",
                suffix="Problem: {input_problem}\nAnalysis:",
                input_variables=["input_problem"],
                example_separator="\n\n"
            )
            
            test_problem = "The application crashes when loading large files"
            formatted_prompt = few_shot_prompt.format(input_problem=test_problem)
            
        else:  # code_review
            example_prompt = PromptTemplate(
                input_variables=["code", "issues", "suggestions", "improved"],
                template="Code:\n{code}\nIssues: {issues}\nSuggestions: {suggestions}\nImproved:\n{improved}"
            )
            
            few_shot_prompt = FewShotPromptTemplate(
                examples=pattern_examples,
                example_prompt=example_prompt,
                prefix="Review the following code and provide improvements:",
                suffix="Code:\n{input_code}\nIssues:",
                input_variables=["input_code"],
                example_separator="\n\n"
            )
            
            test_code = "result = []\nfor i in range(10):\n    if i % 2 == 0:\n        result.append(i * 2)"
            formatted_prompt = few_shot_prompt.format(input_code=test_code)
        
        print(f"Test Input: {test_problem if pattern_name == 'problem_solution' else test_code}")
        print(f"Pattern Prompt:\n{formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}\n")

def task_specific_few_shot():
    """Demonstrate task-specific few-shot examples."""
    print("=== Task-Specific Few-Shot Examples ===")
    
    # Define task-specific examples
    tasks = {
        "translation": {
            "examples": [
                {"english": "Hello, how are you?", "spanish": "Hola, ¿cómo estás?"},
                {"english": "I love programming", "spanish": "Me encanta programar"},
                {"english": "The weather is nice today", "spanish": "El clima está agradable hoy"}
            ],
            "template": "English: {english}\nSpanish: {spanish}"
        },
        "summarization": {
            "examples": [
                {"text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.", "summary": "A pangram containing all alphabet letters."},
                {"text": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science and web development.", "summary": "Python is a simple, readable programming language popular in data science and web development."}
            ],
            "template": "Text: {text}\nSummary: {summary}"
        },
        "question_answering": {
            "examples": [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
                {"question": "What is 2 + 2?", "answer": "4"}
            ],
            "template": "Question: {question}\nAnswer: {answer}"
        }
    }
    
    # Test each task
    for task_name, task_data in tasks.items():
        print(f"\n--- {task_name.upper()} TASK ---")
        
        example_prompt = PromptTemplate(
            input_variables=list(task_data["examples"][0].keys()),
            template=task_data["template"]
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=task_data["examples"],
            example_prompt=example_prompt,
            prefix=f"Perform {task_name}:",
            suffix=f"{task_data['template'].split('{')[0]}{{input}}",
            input_variables=["input"],
            example_separator="\n\n"
        )
        
        # Test inputs for each task
        test_inputs = {
            "translation": "Good morning, everyone",
            "summarization": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions.",
            "question_answering": "What is the largest planet in our solar system?"
        }
        
        test_input = test_inputs[task_name]
        formatted_prompt = few_shot_prompt.format(input=test_input)
        
        print(f"Input: {test_input}")
        print(f"Task-Specific Prompt:\n{formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content}\n")

def main():
    """Run all few-shot learning examples."""
    print("LangChain Few-Shot Learning Examples\n")
    print("=" * 50)
    
    try:
        basic_few_shot_example()
        length_based_example_selection()
        semantic_similarity_example_selection()
        pattern_matching_examples()
        task_specific_few_shot()
        
        print("All few-shot examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 