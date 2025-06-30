"""
Example 09: RunnableBranch in LangChain
======================================

This file demonstrates multiple use cases for RunnableBranch, which routes input to different runnables based on conditions.
"""

from langchain_core.runnables import RunnableLambda, RunnableBranch

# Example 1: Even or odd number
# -----------------------------
def is_even(x):
    return x % 2 == 0

def even_branch(x):
    return f"{x} is even"

def odd_branch(x):
    return f"{x} is odd"

def example_even_odd():
    print("\n[Example 1] Even or odd branch:")
    branch = RunnableBranch(
        (is_even, RunnableLambda(even_branch)),
        (lambda x: True, RunnableLambda(odd_branch)),  # fallback
    )
    print(f"Input: 8, Output: {branch.invoke(8)}")
    print(f"Input: 5, Output: {branch.invoke(5)}")

# Example 2: String length branch
# -------------------------------
def is_short(s):
    return len(s) < 5

def short_branch(s):
    return f"'{s}' is short"

def long_branch(s):
    return f"'{s}' is long"

def example_string_length():
    print("\n[Example 2] String length branch:")
    branch = RunnableBranch(
        (is_short, RunnableLambda(short_branch)),
        (lambda s: True, RunnableLambda(long_branch)),
    )
    print(f"Input: 'cat', Output: {branch.invoke('cat')}")
    print(f"Input: 'elephant', Output: {branch.invoke('elephant')}")

# Example 3: Dictionary key check branch
# --------------------------------------
def has_key_a(d):
    return 'a' in d

def branch_has_a(d):
    return "Has key 'a'"

def branch_no_a(d):
    return "Does not have key 'a'"

def example_dict_branch():
    print("\n[Example 3] Dictionary key check branch:")
    branch = RunnableBranch(
        (has_key_a, RunnableLambda(branch_has_a)),
        (lambda d: True, RunnableLambda(branch_no_a)),
    )
    print(f"Input: {{'a': 1}}, Output: {branch.invoke({'a': 1})}")
    print(f"Input: {{'b': 2}}, Output: {branch.invoke({'b': 2})}")

if __name__ == "__main__":
    example_even_odd()
    example_string_length()
    example_dict_branch() 