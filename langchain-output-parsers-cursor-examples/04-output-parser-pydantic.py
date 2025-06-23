"""
Example 4: Pydantic Output Parser (with Gemini)
This example demonstrates how to use LangChain's PydanticOutputParser to parse Gemini LLM output into a Pydantic model.
"""
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Use Gemini model (same as in test_movie_review_output.py)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Define a Pydantic model for a person
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The person's city")

# Define a prompt that asks for a person's details
prompt = PromptTemplate.from_template("Provide a person's name, age, and city in the format: name: <name> age: <age> city: <city>")

# Create the Pydantic output parser
parser = PydanticOutputParser(pydantic_object=Person)

# Chain: prompt -> Gemini LLM -> parser
chain = prompt | model | parser

# Run the chain and print the result
result = chain.invoke({})
print("Parsed Pydantic model:", result) 