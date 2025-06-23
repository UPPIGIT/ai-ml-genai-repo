"""
Example 17: Travel Itinerary Parser (with Gemini)
Parses a travel itinerary email into a structured itinerary object.
"""
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

class ItineraryItem(BaseModel):
    date: str = Field(description="Date of the activity")
    activity: str = Field(description="Description of the activity")

class TravelItinerary(BaseModel):
    traveler: str = Field(description="Traveler's name")
    items: List[ItineraryItem] = Field(description="List of itinerary items")

prompt = PromptTemplate.from_template(
    """Extract a structured itinerary from this email:
    Hi Alex, your trip is confirmed!
    - June 1: Flight to Paris
    - June 2: Eiffel Tower tour
    - June 3: Louvre Museum visit
    """
)

parser = StructuredOutputParser.from_schema(TravelItinerary)
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed travel itinerary:", result) 