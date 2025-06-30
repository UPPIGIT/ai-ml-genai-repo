"""
Example 12: Advanced RunnablePassthrough - Logging and Conditional Forwarding
============================================================================

This example demonstrates how RunnablePassthrough can be used for logging/debugging in a chain and for conditional data forwarding.
"""

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch

# Example 1: Logging in a chain
# -----------------------------
def log_input(x):
    print(f"[LOG] Input: {x}")
    return x

# Example 2: Conditional forwarding (only forward if number is positive)
def is_positive(x):
    return x > 0

def forward_branch(x):
    return f"Forwarded: {x}"

def block_branch(x):
    return "Blocked: not positive"

if __name__ == "__main__":
    print("\n=== Advanced RunnablePassthrough: Logging in a Chain ===")
    passthrough = RunnablePassthrough()
    logger = RunnableLambda(log_input)
    double = RunnableLambda(lambda x: x * 2)
    # Chain: passthrough -> logger -> double
    result = (passthrough | logger | double).invoke(7)
    print(f"Final Output: {result}")

    print("\n=== Advanced RunnablePassthrough: Conditional Forwarding ===")
    branch = RunnableBranch(
        (is_positive, RunnableLambda(forward_branch)),
        (lambda x: True, RunnableLambda(block_branch)),
    )
    print(f"Input: 5, Output: {branch.invoke(5)}")
    print(f"Input: -3, Output: {branch.invoke(-3)}") 