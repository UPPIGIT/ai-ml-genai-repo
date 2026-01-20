# 05_example_selectors.py
# Example selectors dynamically choose which examples to include in prompts
# This is useful when you have many examples but want to show only the most relevant ones

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector, SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings  # Note: requires API key in practice
from langchain.vectorstores import Chroma

# Example 1: Length-Based Example Selector
# Automatically limits examples based on prompt length to avoid token limits
examples = [
    {"input": "cat", "output": "mammal"},
    {"input": "eagle", "output": "bird"},
    {"input": "salmon", "output": "fish"},
    {"input": "spider", "output": "arachnid"},
    {"input": "frog", "output": "amphibian"},
    {"input": "snake", "output": "reptile"},
]

example_formatter = PromptTemplate(
    input_variables=["input", "output"],
    template="Animal: {input}\nCategory: {output}"
)

# Create length-based selector
# It will include as many examples as fit within max_length
length_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_formatter,
    max_length=100,  # Maximum character length for all examples
)

# Create prompt with the selector
length_prompt = FewShotPromptTemplate(
    example_selector=length_selector,
    example_prompt=example_formatter,
    prefix="Classify each animal:\n",
    suffix="Animal: {input}\nCategory:",
    input_variables=["input"]
)

prompt1 = length_prompt.format(input="dolphin")
print("Example 1 - Length-Based Selector:")
print(prompt1)
print(f"\nCharacter count: {len(prompt1)}")
print("\n" + "="*50 + "\n")

# Example 2: Manual example selector by criteria
# Create a custom selector that chooses examples based on specific rules
class DifficultyBasedSelector:
    """Select examples based on difficulty level"""
    
    def __init__(self, examples):
        self.examples = examples
    
    def select_examples(self, input_variables):
        """Select examples matching the difficulty"""
        difficulty = input_variables.get("difficulty", "medium")
        # Filter examples by difficulty
        selected = [ex for ex in self.examples if ex.get("difficulty") == difficulty]
        return selected[:3]  # Return up to 3 examples

# Examples with difficulty metadata
coding_examples = [
    {
        "difficulty": "easy",
        "problem": "Print 'Hello World'",
        "solution": "print('Hello World')"
    },
    {
        "difficulty": "easy",
        "problem": "Add two numbers",
        "solution": "def add(a, b): return a + b"
    },
    {
        "difficulty": "medium",
        "problem": "Reverse a string",
        "solution": "def reverse(s): return s[::-1]"
    },
    {
        "difficulty": "medium",
        "problem": "Find factorial",
        "solution": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    },
    {
        "difficulty": "hard",
        "problem": "Implement binary search",
        "solution": "def binary_search(arr, x): ..."
    }
]

difficulty_selector = DifficultyBasedSelector(coding_examples)

# Simulate selecting easy examples
easy_examples = difficulty_selector.select_examples({"difficulty": "easy"})
print("Example 2 - Custom Difficulty Selector (Easy):")
for ex in easy_examples:
    print(f"Problem: {ex['problem']}")
    print(f"Solution: {ex['solution']}\n")
print("="*50 + "\n")

# Example 3: Random Example Selector
# Useful for variety in responses or A/B testing
import random

class RandomExampleSelector:
    """Randomly select N examples"""
    
    def __init__(self, examples, n=2):
        self.examples = examples
        self.n = n
    
    def select_examples(self, input_variables):
        """Randomly select n examples"""
        return random.sample(self.examples, min(self.n, len(self.examples)))

joke_examples = [
    {"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side!"},
    {"setup": "What do you call a bear with no teeth?", "punchline": "A gummy bear!"},
    {"setup": "Why don't scientists trust atoms?", "punchline": "Because they make up everything!"},
    {"setup": "What did the ocean say to the beach?", "punchline": "Nothing, it just waved!"},
]

random_selector = RandomExampleSelector(joke_examples, n=2)
selected_jokes = random_selector.select_examples({})

print("Example 3 - Random Selector:")
for joke in selected_jokes:
    print(f"Setup: {joke['setup']}")
    print(f"Punchline: {joke['punchline']}\n")
print("="*50 + "\n")

# Example 4: Conditional Example Selector
# Select examples based on input characteristics
class ConditionalSelector:
    """Select examples based on input type"""
    
    def __init__(self, examples):
        self.examples = examples
    
    def select_examples(self, input_variables):
        user_input = input_variables.get("input", "")
        
        # If input contains numbers, show math examples
        if any(char.isdigit() for char in user_input):
            return [ex for ex in self.examples if ex["type"] == "math"][:2]
        # If input contains letters only, show text examples
        else:
            return [ex for ex in self.examples if ex["type"] == "text"][:2]

mixed_examples = [
    {"type": "math", "input": "2 + 2", "output": "4"},
    {"type": "math", "input": "10 * 5", "output": "50"},
    {"type": "text", "input": "hello", "output": "HELLO"},
    {"type": "text", "input": "world", "output": "WORLD"},
]

conditional_selector = ConditionalSelector(mixed_examples)

# Test with numeric input
numeric_examples = conditional_selector.select_examples({"input": "25 + 13"})
print("Example 4 - Conditional Selector (Numeric Input):")
for ex in numeric_examples:
    print(f"Input: {ex['input']} -> Output: {ex['output']}")

print()

# Test with text input
text_examples = conditional_selector.select_examples({"input": "python"})
print("Conditional Selector (Text Input):")
for ex in text_examples:
    print(f"Input: {ex['input']} -> Output: {ex['output']}")

print("\n" + "="*50 + "\n")

# Example 5: Hybrid selector combining multiple strategies
class HybridSelector:
    """Combines length limits with relevance"""
    
    def __init__(self, examples, max_examples=3):
        self.examples = examples
        self.max_examples = max_examples
    
    def select_examples(self, input_variables):
        user_input = input_variables.get("input", "").lower()
        
        # Score examples by keyword relevance
        scored_examples = []
        for ex in self.examples:
            score = sum(1 for word in ex["keywords"] if word in user_input)
            scored_examples.append((score, ex))
        
        # Sort by relevance and limit by max_examples
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        return [ex for score, ex in scored_examples[:self.max_examples]]

keyword_examples = [
    {"keywords": ["web", "html", "css"], "topic": "Web Development", "info": "Build websites"},
    {"keywords": ["data", "analysis", "pandas"], "topic": "Data Science", "info": "Analyze data"},
    {"keywords": ["machine", "learning", "model"], "topic": "ML", "info": "Train models"},
    {"keywords": ["database", "sql", "query"], "topic": "Databases", "info": "Store data"},
]

hybrid_selector = HybridSelector(keyword_examples, max_examples=2)
selected = hybrid_selector.select_examples({"input": "I want to learn about data analysis and pandas"})

print("Example 5 - Hybrid Selector:")
for ex in selected:
    print(f"Topic: {ex['topic']} - {ex['info']}")