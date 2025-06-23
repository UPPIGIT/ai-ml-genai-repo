"""
Example 3: JSON Output Parser (with Gemini)
This example demonstrates how to use LangChain's JsonOutputParser to parse a JSON object from a Gemini LLM output.
"""
from langchain.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Use Gemini model (same as in test_movie_review_output.py)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Define a prompt that asks for a JSON object
prompt = PromptTemplate.from_template("Provide a JSON object with name, age, and city.")

# Create the JSON output parser
parser = JsonOutputParser()

# Chain: prompt -> Gemini LLM -> parser
chain = prompt | model | parser

# Run the chain and print the result
result = chain.invoke({})
print("Parsed JSON:", result) 