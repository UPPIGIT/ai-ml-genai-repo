"""
Example 1: Basic String Output Parser (with Gemini)
This example demonstrates how to use LangChain's StrOutputParser to parse the output of a Gemini LLM as a simple string.
"""
from langchain.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Use Gemini model (same as in test_movie_review_output.py)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Define a simple prompt
prompt = PromptTemplate.from_template("Say hello!")

# Create the string output parser
parser = StrOutputParser()

# Chain: prompt -> Gemini LLM -> parser
chain = prompt | model | parser

# Run the chain and print the result
result = chain.invoke({})
print("Parsed output:", result) 