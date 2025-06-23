from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Initialize the Anthropic chat model
chat_model = ChatAnthropic(model="claude-3-7-sonnet-20250219")
# Invoke the chat model with a prompt
result = chat_model.invoke("What is the capital of India?")
print(result)
print(result.content)
