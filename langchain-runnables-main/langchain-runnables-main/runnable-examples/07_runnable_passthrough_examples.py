"""
Example 07: RunnablePassthrough in LangChain
===========================================

This file demonstrates multiple use cases for RunnablePassthrough, which simply returns its input unchanged. Useful for chaining and debugging.
"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Example 1: Pass through a string
# --------------------------------
def example_passthrough_string():
    print("\n[Example 1] Passthrough string:")
    passthrough = RunnablePassthrough()
    print(f"Input: 'hello', Output: {passthrough.invoke('hello')}")

# Example 2: Pass through a dictionary
# ------------------------------------
def example_passthrough_dict():
    print("\n[Example 2] Passthrough dictionary:")
    passthrough = RunnablePassthrough()
    print(f"Input: {{'foo': 123}}, Output: {passthrough.invoke({'foo': 123})}")

# Example 3: Use in a chain
# -------------------------
def add_exclamation(s):
    return s + '!'

def example_passthrough_chain():
    print("\n[Example 3] Passthrough in a chain:")
    passthrough = RunnablePassthrough()
    add_excl = RunnableLambda(add_exclamation)
    # Chain: passthrough -> add_exclamation
    result = (passthrough | add_excl).invoke('LangChain')
    print(f"Input: 'LangChain', Output: {result}")

if __name__ == "__main__":
    example_passthrough_string()
    example_passthrough_dict()
    example_passthrough_chain() 