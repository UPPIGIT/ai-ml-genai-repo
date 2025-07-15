"""
Multiple Tool Call Example in LangChain
--------------------------------------
This script demonstrates how to define and use multiple tools in sequence.
It shows how to create a simple workflow where the output of one tool is used as input to another.
"""

from langchain.tools import Tool

def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers.
    Args:
        a (int): First number.
        b (int): Second number.
    Returns:
        int: Sum of a and b.
    """
    return a + b

def multiply_by_two(x: int) -> int:
    """
    Multiplies the input by two.
    Args:
        x (int): Input number.
    Returns:
        int: Result after multiplication.
    """
    return x * 2

# Create Tool instances
add_tool = Tool(
    name="add_numbers",
    func=add_numbers,
    description="Adds two integers."
)
multiply_tool = Tool(
    name="multiply_by_two",
    func=multiply_by_two,
    description="Multiplies an integer by two."
)

if __name__ == "__main__":
    # Call the tools in sequence
    sum_result = add_tool.run(3, 4)  # 3 + 4 = 7
    final_result = multiply_tool.run(sum_result)  # 7 * 2 = 14
    print(f"(3 + 4) * 2 = {final_result}")  # Output: (3 + 4) * 2 = 14 