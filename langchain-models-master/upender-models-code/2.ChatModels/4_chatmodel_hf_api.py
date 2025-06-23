from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Initialize the Hugging Face chat model
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
                                                     task="text-generation")
chat_model = ChatHuggingFace(llm=llm)
# Invoke the chat model with a prompt
result = chat_model.invoke("What is the capital of India?")
print(result.content)

    