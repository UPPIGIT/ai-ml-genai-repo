# 04_few_shot_prompts.py
# Few-shot prompting provides examples to guide the AI's responses
# This is like teaching by showing examples before asking for the real task

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Example 1: Basic few-shot with static examples
# Define the format for each example
example_formatter = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Create a list of examples (this is our "training data")
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
]

# Create the few-shot prompt
few_shot_prompt1 = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_formatter,
    prefix="Give the antonym of each word:",  # Instruction before examples
    suffix="Input: {word}\nOutput:",  # What comes after examples
    input_variables=["word"]
)

prompt1 = few_shot_prompt1.format(word="big")
print("Example 1 - Basic Few-Shot:")
print(prompt1)
print("\n" + "="*50 + "\n")

# Example 2: Few-shot for code generation
# Teaching the AI to write functions by example
code_example_formatter = PromptTemplate(
    input_variables=["description", "code"],
    template="Description: {description}\n```python\n{code}\n```"
)

code_examples = [
    {
        "description": "Function to add two numbers",
        "code": "def add(a, b):\n    return a + b"
    },
    {
        "description": "Function to check if number is even",
        "code": "def is_even(n):\n    return n % 2 == 0"
    }
]

code_prompt = FewShotPromptTemplate(
    examples=code_examples,
    example_prompt=code_example_formatter,
    prefix="Generate Python functions based on descriptions:\n",
    suffix="Description: {description}\n```python\n",
    input_variables=["description"]
)

prompt2 = code_prompt.format(description="Function to calculate factorial")
print("Example 2 - Code Generation Few-Shot:")
print(prompt2)
print("\n" + "="*50 + "\n")

# Example 3: Few-shot for structured data extraction
# Teaching the AI to extract specific information
extraction_formatter = PromptTemplate(
    input_variables=["text", "result"],
    template="Text: {text}\nExtracted: {result}"
)

extraction_examples = [
    {
        "text": "John lives in New York and works as a teacher.",
        "result": "Name: John, Location: New York, Job: teacher"
    },
    {
        "text": "Sarah is a doctor in London.",
        "result": "Name: Sarah, Location: London, Job: doctor"
    }
]

extraction_prompt = FewShotPromptTemplate(
    examples=extraction_examples,
    example_prompt=extraction_formatter,
    prefix="Extract name, location, and job from text:\n",
    suffix="Text: {text}\nExtracted:",
    input_variables=["text"]
)

prompt3 = extraction_prompt.format(
    text="Mike is an engineer in San Francisco."
)
print("Example 3 - Data Extraction Few-Shot:")
print(prompt3)
print("\n" + "="*50 + "\n")

# Example 4: Dynamic example selection
# Sometimes you want to include only relevant examples based on input
# This uses example_selector (we'll simulate it here)

math_examples = [
    {"operation": "addition", "problem": "2 + 3", "answer": "5"},
    {"operation": "subtraction", "problem": "10 - 4", "answer": "6"},
    {"operation": "multiplication", "problem": "5 * 6", "answer": "30"},
    {"operation": "division", "problem": "20 / 4", "answer": "5"},
]

math_formatter = PromptTemplate(
    input_variables=["problem", "answer"],
    template="Problem: {problem} = {answer}"
)

# For addition, we might want to show only addition examples
addition_examples = [ex for ex in math_examples if ex["operation"] == "addition"]

math_prompt = FewShotPromptTemplate(
    examples=addition_examples,  # Filtered examples
    example_prompt=math_formatter,
    prefix="Solve the following math problems:\n",
    suffix="Problem: {problem} = ",
    input_variables=["problem"]
)

prompt4 = math_prompt.format(problem="7 + 8")
print("Example 4 - Filtered Examples:")
print(prompt4)
print("\n" + "="*50 + "\n")

# Example 5: Few-shot with complex formatting
# Using few-shot for JSON output formatting
json_formatter = PromptTemplate(
    input_variables=["sentence", "json"],
    template='Sentence: "{sentence}"\nJSON: {json}'
)

json_examples = [
    {
        "sentence": "The movie was amazing!",
        "json": '{"sentiment": "positive", "intensity": "high"}'
    },
    {
        "sentence": "It was okay.",
        "json": '{"sentiment": "neutral", "intensity": "low"}'
    }
]

json_prompt = FewShotPromptTemplate(
    examples=json_examples,
    example_prompt=json_formatter,
    prefix="Convert sentences to sentiment JSON:\n",
    suffix='Sentence: "{sentence}"\nJSON:',
    input_variables=["sentence"]
)

prompt5 = json_prompt.format(sentence="I hated it!")
print("Example 5 - JSON Output Few-Shot:")
print(prompt5)