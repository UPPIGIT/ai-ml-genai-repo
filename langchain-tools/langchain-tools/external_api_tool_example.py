"""
External API Tool Example in LangChain
-------------------------------------
This script demonstrates how to wrap an external API call as a LangChain tool.
For demonstration, it uses the public JSONPlaceholder API to fetch user data.
"""

import requests
from langchain.tools import Tool

def fetch_user_data(user_id: int) -> dict:
    """
    Fetches user data from the JSONPlaceholder API.
    Args:
        user_id (int): The user ID to fetch.
    Returns:
        dict: User data as a dictionary.
    """
    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

fetch_user_tool = Tool(
    name="fetch_user_data",
    func=fetch_user_data,
    description="Fetches user data from JSONPlaceholder API by user ID."
)

if __name__ == "__main__":
    user = fetch_user_tool.run(1)
    print(f"User 1 data: {user}") 