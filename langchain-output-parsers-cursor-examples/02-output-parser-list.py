"""
Example 2: Comma-Separated List Output Parser (with Gemini)
This example demonstrates how to use LangChain's CommaSeparatedListOutputParser to parse a comma-separated list from a Gemini LLM output.
"""
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Use Gemini model (same as in test_movie_review_output.py)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Define a prompt that asks for a list of fruits
prompt = PromptTemplate.from_template("List three fruits.")

# Create the comma-separated list output parser
parser = CommaSeparatedListOutputParser()

# Chain: prompt -> Gemini LLM -> parser
chain = prompt | model | parser

# Run the chain and print the result
result = chain.invoke({})
print("Parsed list:", result) 