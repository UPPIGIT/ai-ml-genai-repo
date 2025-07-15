"""
Single Tool Call Example in LangChain
-------------------------------------
This script demonstrates how to use a single custom tool with LangChain's Tool class.
It shows how to define a function-based tool and call it directly.
"""

from langchain.tools import Tool

def square_number(number: int) -> int:
    """
    Returns the square of the given number.
    Args:
        number (int): The number to square.
    Returns:
        int: The squared value.
    """
    return number * number

# Create a Tool instance from the function
tool = Tool(
    name="square_number",
    func=square_number,
    description="Returns the square of a given integer."
)

if __name__ == "__main__":
    # Call the tool directly
    result = tool.run(5)
    print(f"The square of 5 is {result}")  # Output: The square of 5 is 25 