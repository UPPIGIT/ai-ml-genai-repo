"""
Example 06: RunnableParallel in LangChain
========================================

This file demonstrates multiple use cases for RunnableParallel, which runs several runnables in parallel and collects their outputs.
"""

from langchain_core.runnables import RunnableLambda, RunnableParallel

# Example 1: Parallel math operations
# -----------------------------------
def add(x):
    return x + 1

def square(x):
    return x * x

def example_parallel_math():
    print("\n[Example 1] Parallel math operations:")
    parallel = RunnableParallel({
        "add": RunnableLambda(add),
        "square": RunnableLambda(square),
    })
    print(f"Input: 3, Output: {parallel.invoke(3)}")
    # Output: {'add': 4, 'square': 9}

# Example 2: Parallel string operations
# -------------------------------------
def upper(s):
    return s.upper()

def lower(s):
    return s.lower()

def example_parallel_strings():
    print("\n[Example 2] Parallel string operations:")
    parallel = RunnableParallel({
        "upper": RunnableLambda(upper),
        "lower": RunnableLambda(lower),
    })
    print(f"Input: 'LangChain', Output: {parallel.invoke('LangChain')}")
    # Output: {'upper': 'LANGCHAIN', 'lower': 'langchain'}

# Example 3: Parallel dictionary extraction
# ----------------------------------------
def get_keys(d):
    return list(d.keys())

def get_values(d):
    return list(d.values())

def example_parallel_dict():
    print("\n[Example 3] Parallel dictionary extraction:")
    parallel = RunnableParallel({
        "keys": RunnableLambda(get_keys),
        "values": RunnableLambda(get_values),
    })
    print(f"Input: {{'a': 1, 'b': 2}}, Output: {parallel.invoke({'a': 1, 'b': 2})}")
    # Output: {'keys': ['a', 'b'], 'values': [1, 2]}

if __name__ == "__main__":
    example_parallel_math()
    example_parallel_strings()
    example_parallel_dict() 