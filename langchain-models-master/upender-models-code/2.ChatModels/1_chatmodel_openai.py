from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAI chat model
chat_model = ChatOpenAI(model="gpt-4", temperature=0.7,max_completion_tokens=10)

# Invoke the chat model with a prompt
result =chat_model.invoke("what is the capital of India?")
print (result)

print(result.content)
