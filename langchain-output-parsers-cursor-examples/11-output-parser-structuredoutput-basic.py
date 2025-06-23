"""
Example 11: StructuredOutputParser with a Simple Pydantic Schema (with Gemini)
This example demonstrates how to use StructuredOutputParser to parse Gemini output into a structured Python object.
"""
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Define a simple Pydantic schema
class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD")
    in_stock: bool = Field(description="Is the product in stock?")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Prompt instructs Gemini to return data in the schema format
prompt = PromptTemplate.from_template(
    "Provide details for a product called 'SuperWidget' with a price of 19.99 USD and in stock. Respond in JSON."
)

parser = StructuredOutputParser.from_schema(Product)
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed structured output:", result) 