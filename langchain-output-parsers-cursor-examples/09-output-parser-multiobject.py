"""
Example 9: Multi-Object Output Parser (List of JSON Objects, with Gemini)
This example demonstrates parsing a list of objects from Gemini output.
"""
from langchain.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
prompt = PromptTemplate.from_template(
    "Provide a JSON array of three people, each with name and age."
)
parser = JsonOutputParser()

chain = prompt | model | parser

result = chain.invoke({})
print("Parsed list of people:", result) 