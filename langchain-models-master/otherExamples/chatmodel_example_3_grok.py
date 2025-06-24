import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
llm = ChatGroq(model_name="llama3-8b-8192")

response = llm.invoke([
    HumanMessage(content="Hello, who are you?")
])
print("Groq response:", response.content) 