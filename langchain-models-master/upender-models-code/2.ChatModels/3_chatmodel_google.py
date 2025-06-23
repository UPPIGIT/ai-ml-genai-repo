from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Initialize the Google Generative AI chat model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.7, max_completion_tokens=10)
# Invoke the chat model with a prompt
result = chat_model.invoke("who is prime minister of india?")
print(result.content)
