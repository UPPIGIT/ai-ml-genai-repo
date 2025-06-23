from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# TravelExperience schema
class TravelExperience(BaseModel):
    location: str = Field(description="The destination name")
    country: Optional[str] = Field(default=None, description="The country of the destination")
    highlights: Optional[list[str]] = Field(default=None, description="Highlights or must-see attractions")
    summary: str = Field(description="A brief summary of the travel experience")
    rating: float = Field(description="A rating from 1 to 10")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Overall sentiment: positive, negative, or neutral")
    tips: Optional[list[str]] = Field(default=None, description="Travel tips or advice")
    reviewer: Optional[str] = Field(default=None, description="Name of the reviewer")
    visited_date: Optional[str] = Field(default=None, description="Date or month/year of visit")

structured_model = model.with_structured_output(TravelExperience)

# Example travel experience
#result = structured_model.invoke("""I visited Kyoto, Japan in April 2023, and it was an unforgettable experience! The cherry blossoms were in full bloom, making every park and temple look magical. Highlights included visiting Fushimi Inari Shrine, exploring the Arashiyama Bamboo Grove, and enjoying traditional kaiseki cuisine. The city is clean, safe, and easy to navigate. My only regret is not spending more time there!\n\nTips:\nVisit during cherry blossom season for the best views\nBuy a day pass for public transport\nTry local street food at Nishiki Market\n\nReview by Emily Chen\nVisited: April 2023\nRating: 9.8/10\n""")

my_review ="""I visited Hyderabad, India in December 2023, and it was a vibrant and culturally rich experience! The city perfectly blends ancient history with modern development. Highlights included exploring the majestic Golconda Fort, admiring the intricate architecture of Charminar, and savoring the world-famous Hyderabadi biryani at Paradise. The bazaars around Laad Bazaar were full of colorful bangles, pearls, and local crafts.

Hyderabad is known for its friendly locals, historical landmarks, and delicious cuisine. The weather in December was pleasantly cool, making sightseeing enjoyable. My only regret was not making time to visit Ramoji Film City!

Tips:

Try authentic Hyderabadi biryani at local restaurants

Visit Golconda Fort in the evening for the light and sound show

Use ride-hailing apps for convenient local travel

Review by: Arjun Mehta
Visited: December 2023
Rating: 9.4/10



"""
result = structured_model.invoke(my_review)
print(result) 