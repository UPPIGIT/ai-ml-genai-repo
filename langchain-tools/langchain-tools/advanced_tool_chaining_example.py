"""
Advanced Tool Chaining Example in LangChain
------------------------------------------
This script demonstrates advanced tool chaining using LangChain.
It shows how to create a custom tool that internally calls other tools and applies custom logic.
"""

from langchain.tools import Tool, BaseTool

# Define basic tools
def get_length(text: str) -> int:
    """
    Returns the length of the input text.
    """
    return len(text)

def is_even(number: int) -> bool:
    """
    Returns True if the number is even, False otherwise.
    """
    return number % 2 == 0

length_tool = Tool(
    name="get_length",
    func=get_length,
    description="Returns the length of a string."
)
even_tool = Tool(
    name="is_even",
    func=is_even,
    description="Checks if a number is even."
)

# Advanced tool that chains the above tools
class TextLengthEvenChecker(BaseTool):
    """
    Custom tool that checks if the length of a given text is even.
    Demonstrates chaining and custom logic.
    """
    name = "text_length_even_checker"
    description = "Checks if the length of the input text is even."

    def _run(self, text: str) -> str:
        length = length_tool.run(text)
        even = even_tool.run(length)
        if even:
            return f"The length of '{text}' is {length}, which is even."
        else:
            return f"The length of '{text}' is {length}, which is odd."

if __name__ == "__main__":
    checker = TextLengthEvenChecker()
    print(checker.run("LangChain"))  # Output: The length of 'LangChain' is 9, which is odd.
    print(checker.run("Tools"))      # Output: The length of 'Tools' is 5, which is odd.
    print(checker.run("Hello!"))     # Output: The length of 'Hello!' is 6, which is even. 