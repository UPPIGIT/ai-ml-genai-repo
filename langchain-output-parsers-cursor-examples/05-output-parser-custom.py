"""
Example 5: Custom Output Parser (Advanced, with Gemini)
This example demonstrates how to create and use a custom output parser in LangChain to extract structured data from free-form Gemini LLM output.
"""
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()

# Use Gemini model (same as in test_movie_review_output.py)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Define a custom output parser that extracts name, age, and city from free text
class PersonInfoParser(BaseOutputParser):
    def parse(self, text: str):
        # Use regex to extract fields from the text
        name_match = re.search(r"Name: (.+)", text)
        age_match = re.search(r"Age: (\\d+)", text)
        city_match = re.search(r"City: (.+)", text)
        return {
            "name": name_match.group(1).strip() if name_match else None,
            "age": int(age_match.group(1)) if age_match else None,
            "city": city_match.group(1).strip() if city_match else None,
        }

# Define a prompt that asks for a person's details in free text
prompt = PromptTemplate.from_template("Tell me about a person named Alice, including her age and city.")

# Create the custom output parser
parser = PersonInfoParser()

# Chain: prompt -> Gemini LLM -> parser
chain = prompt | model | parser

# Run the chain and print the result
result = chain.invoke({})
print("Parsed structured data:", result) 