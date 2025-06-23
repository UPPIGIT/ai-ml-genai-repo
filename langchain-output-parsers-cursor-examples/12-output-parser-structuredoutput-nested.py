"""
Example 12: StructuredOutputParser with a Nested Pydantic Schema (with Gemini)
This example demonstrates how to use StructuredOutputParser to parse nested structured data from Gemini output.
"""
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# Define nested Pydantic schemas
class Review(BaseModel):
    reviewer: str = Field(description="Name of the reviewer")
    rating: int = Field(description="Rating out of 5")
    comment: str = Field(description="Review comment")

class ProductWithReviews(BaseModel):
    name: str = Field(description="Product name")
    reviews: List[Review] = Field(description="List of product reviews")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt = PromptTemplate.from_template(
    """Provide details for a product called 'SuperWidget' and two reviews in JSON format.\nEach review should have a reviewer name, rating (1-5), and a comment."""
)

parser = StructuredOutputParser.from_schema(ProductWithReviews)
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed nested structured output:", result) 