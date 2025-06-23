from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAI model
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
# Invoke the model with a prompt
# This is a simple example of using the OpenAI model to get a response
llm_response = llm.invoke("What is the capital of France?")
print(llm_response)
