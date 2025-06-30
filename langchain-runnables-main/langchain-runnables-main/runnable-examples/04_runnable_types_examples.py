"""
Example 04: Runnable Types in LangChain
======================================

This file demonstrates the usage of various core Runnable types in LangChain:
- RunnableLambda
- RunnableParallel
- RunnablePassthrough
- RunnableSequence
- RunnableBranch

Each section is self-contained and includes comments for clarity.
"""

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    RunnableBranch,
)

# 1. RunnableLambda Example
# -------------------------
def double(x):
    """Simple function to double the input."""
    return x * 2

def lambda_example():
    print("\n=== RunnableLambda Example ===")
    double_runnable = RunnableLambda(double)
    result = double_runnable.invoke(5)
    print(f"Input: 5, Output: {result}")

# 2. RunnableParallel Example
# ---------------------------
def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def parallel_example():
    print("\n=== RunnableParallel Example ===")
    parallel = RunnableParallel({
        "double": RunnableLambda(double),
        "square": RunnableLambda(square),
        "cube": RunnableLambda(cube),
    })
    result = parallel.invoke(3)
    print(f"Input: 3, Output: {result}")
    # Output: {'double': 6, 'square': 9, 'cube': 27}

# 3. RunnablePassthrough Example
# ------------------------------
def passthrough_example():
    print("\n=== RunnablePassthrough Example ===")
    passthrough = RunnablePassthrough()
    result = passthrough.invoke({"foo": "bar", "num": 42})
    print(f"Input: {{'foo': 'bar', 'num': 42}}, Output: {result}")

# 4. RunnableSequence Example
# ---------------------------
def add_one(x):
    return x + 1

def multiply_by_ten(x):
    return x * 10

def sequence_example():
    print("\n=== RunnableSequence Example ===")
    sequence = RunnableSequence([
        RunnableLambda(add_one),
        RunnableLambda(multiply_by_ten),
    ])
    result = sequence.invoke(4)
    print(f"Input: 4, Output: {result}")
    # Output: (4 + 1) * 10 = 50

# 5. RunnableBranch Example
# -------------------------
def branch_example():
    print("\n=== RunnableBranch Example ===")
    def is_even(x):
        return x % 2 == 0
    def even_branch(x):
        return f"{x} is even"
    def odd_branch(x):
        return f"{x} is odd"
    branch = RunnableBranch(
        (is_even, RunnableLambda(even_branch)),
        (lambda x: True, RunnableLambda(odd_branch)),  # fallback
    )
    print(f"Input: 6, Output: {branch.invoke(6)}")
    print(f"Input: 7, Output: {branch.invoke(7)}")

if __name__ == "__main__":
    lambda_example()
    parallel_example()
    passthrough_example()
    sequence_example()
    branch_example() 