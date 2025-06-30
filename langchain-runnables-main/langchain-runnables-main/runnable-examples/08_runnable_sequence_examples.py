"""
Example 08: RunnableSequence in LangChain
========================================

This file demonstrates multiple use cases for RunnableSequence, which chains runnables in a specific order.
"""

from langchain_core.runnables import RunnableLambda, RunnableSequence

# Example 1: Math sequence
# ------------------------
def add_two(x):
    return x + 2

def times_five(x):
    return x * 5

def example_sequence_math():
    print("\n[Example 1] Math sequence:")
    sequence = RunnableSequence([
        RunnableLambda(add_two),
        RunnableLambda(times_five),
    ])
    print(f"Input: 3, Output: {sequence.invoke(3)}")
    # Output: (3 + 2) * 5 = 25

# Example 2: String transformation sequence
# -----------------------------------------
def strip_spaces(s):
    return s.strip()

def to_upper(s):
    return s.upper()

def example_sequence_string():
    print("\n[Example 2] String transformation sequence:")
    sequence = RunnableSequence([
        RunnableLambda(strip_spaces),
        RunnableLambda(to_upper),
    ])
    print(f"Input: '  hello  ', Output: {sequence.invoke('  hello  ')}")
    # Output: 'HELLO'

# Example 3: Dictionary processing sequence
# ----------------------------------------
def get_items(d):
    return list(d.items())

def count_items(items):
    return len(items)

def example_sequence_dict():
    print("\n[Example 3] Dictionary processing sequence:")
    sequence = RunnableSequence([
        RunnableLambda(get_items),
        RunnableLambda(count_items),
    ])
    print(f"Input: {{'a': 1, 'b': 2}}, Output: {sequence.invoke({'a': 1, 'b': 2})}")
    # Output: 2

if __name__ == "__main__":
    example_sequence_math()
    example_sequence_string()
    example_sequence_dict() 