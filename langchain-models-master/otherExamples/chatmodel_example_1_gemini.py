import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

response = llm.invoke([
    HumanMessage(content="who is Virat kohli")
])
print("Gemini response:", response.content) 