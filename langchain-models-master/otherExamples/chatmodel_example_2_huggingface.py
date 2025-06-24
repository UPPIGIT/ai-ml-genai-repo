import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()



llm=HuggingFaceEndpoint(repo_id="mistralai/Magistral-Small-2506",
                                                     task="text-generation")
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("who is prime minister of india?")
print("HuggingFace response:", response.content) 