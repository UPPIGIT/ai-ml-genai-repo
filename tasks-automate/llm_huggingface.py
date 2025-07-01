from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Initialize the Hugging Face chat model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
chat_model = ChatHuggingFace(llm=llm)

# Invoke the chat model with a prompt
def main():
    result = chat_model.invoke("What is the capital of India?")
    print(result.content)

if __name__ == "__main__":
    main() 