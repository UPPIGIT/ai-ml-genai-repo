"""
Example 05: RunnableLambda in LangChain
======================================

This file demonstrates multiple use cases for RunnableLambda, which wraps a Python function as a runnable component in a LangChain pipeline.
"""

from langchain_core.runnables import RunnableLambda

# Example 1: Double a number
# --------------------------
def double(x):
    """Returns double the input value."""
    return x * 2

def example_double():
    print("\n[Example 1] Double a number:")
    r = RunnableLambda(double)
    print(f"Input: 4, Output: {r.invoke(4)}")

# Example 2: String reversal
# --------------------------
def reverse_string(s):
    """Returns the reversed string."""
    return s[::-1]

def example_reverse():
    print("\n[Example 2] Reverse a string:")
    r = RunnableLambda(reverse_string)
    print(f"Input: 'hello', Output: {r.invoke('hello')}")

# Example 3: Dictionary transformation
# ------------------------------------
def extract_keys(d):
    """Returns the keys of a dictionary as a list."""
    return list(d.keys())

def example_dict_keys():
    print("\n[Example 3] Extract dictionary keys:")
    r = RunnableLambda(extract_keys)
    print(f"Input: {{'a': 1, 'b': 2}}, Output: {r.invoke({'a': 1, 'b': 2})}")

# Example 4: Custom logic with lambda
# -----------------------------------
example_lambda = RunnableLambda(lambda x: f"Length is {len(x)}")

def example_lambda_func():
    print("\n[Example 4] Lambda function for string length:")
    print(f"Input: 'LangChain', Output: {example_lambda.invoke('LangChain')}")

if __name__ == "__main__":
    example_double()
    example_reverse()
    example_dict_keys()
    example_lambda_func() 