from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Initialize the Groq chat model
chat_model = ChatGroq(model="llama3-8b-8192")

# Invoke the chat model with a prompt
def main():
    result = chat_model.invoke("who is prime minister of india?")
    print(result.content)

if __name__ == "__main__":
    main() 