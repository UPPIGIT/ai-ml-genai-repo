from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Initialize the Google Generative AI chat model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Invoke the chat model with a prompt
def main():
    result = chat_model.invoke("who is prime minister of india?")
    print(result.content)

if __name__ == "__main__":
    main() 