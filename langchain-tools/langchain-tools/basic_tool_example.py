"""
Basic LangChain Tool Example
---------------------------
This script demonstrates how to define a simple custom tool in LangChain.
A tool in LangChain is a callable object (usually a function or class) that can be invoked by an agent or chain.
"""

from langchain.tools import BaseTool

class HelloWorldTool(BaseTool):
    """
    A simple tool that returns a greeting message.
    Inherit from BaseTool and implement the _run method for synchronous execution.
    """
    name = "hello_world"
    description = "Returns a friendly greeting."

    def _run(self, name: str) -> str:
        """
        Synchronous execution method for the tool.
        Args:
            name (str): The name to greet.
        Returns:
            str: Greeting message.
        """
        return f"Hello, {name}! Welcome to LangChain tools."

# Example usage
if __name__ == "__main__":
    tool = HelloWorldTool()
    print(tool.run("Alice"))  # Output: Hello, Alice! Welcome to LangChain tools. 