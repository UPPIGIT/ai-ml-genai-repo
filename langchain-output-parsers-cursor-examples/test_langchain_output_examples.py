from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

# Example 1: Positive review (Laptop)
result1 = structured_model.invoke("""I recently purchased the Dell XPS 15, and it has exceeded my expectations. The 4K display is breathtaking, and the build quality feels premium. Performance is top-notch for both work and light gaming. Battery life is solid, lasting me through a full workday. The keyboard is comfortable, and the trackpad is very responsive. The speakers are surprisingly good for a laptop.\n\nPros:\nStunning 4K display\nExcellent build quality\nGreat performance\nLong battery life\n\nReview by Priya Sharma\n""")
print("Positive Review Result:", result1)

# Example 2: Negative review (Headphones)
result2 = structured_model.invoke("""I was disappointed with the new XYZ Wireless Headphones. The sound quality is mediocre, with weak bass and tinny highs. The Bluetooth connection drops frequently, and the ear cushions are uncomfortable after an hour. The battery life is much shorter than advertised. For the price, I expected much better.\n\nCons:\nPoor sound quality\nUnreliable Bluetooth connection\nUncomfortable fit\nShort battery life\n\nReview by Alex Kim\n""")
print("Negative Review Result:", result2)

# Example 3: Neutral review (Coffee Maker)
result3 = structured_model.invoke("""The BrewMaster 3000 coffee maker does its job, but nothing stands out. It makes decent coffee, but the machine is a bit noisy. Setup was straightforward, and cleaning is easy. The design is simple and fits well in my kitchen. However, the water reservoir is small, so I have to refill it often.\n\nPros:\nEasy to use\nSimple design\nEasy to clean\n\nCons:\nNoisy operation\nSmall water reservoir\n\nReview by Jamie Lee\n""")
print("Neutral Review Result:", result3) 