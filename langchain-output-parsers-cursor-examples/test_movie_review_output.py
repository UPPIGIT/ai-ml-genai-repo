from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

#model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#gemma-3-4b-it
#model = ChatGoogleGenerativeAI(model="gemma-3-4b-it")
#gemini-2.5-pro-preview-05-06
#model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06")
#gemini-2.5-flash
#model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#gemini-2.5-flash-lite-preview-06-17
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# MovieReview schema
class MovieReview(BaseModel):
    title: str = Field(description="The movie's title")
    director: Optional[str] = Field(default=None, description="The director's name")
    main_actors: Optional[list[str]] = Field(default=None, description="List of main actors")
    key_themes: Optional[list[str]] = Field(default=None, description="Key themes in the movie")
    summary: str = Field(description="A brief summary of the review")
    rating: float = Field(description="A rating from 1 to 10")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Overall sentiment: positive, negative, or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Positive aspects of the movie")
    cons: Optional[list[str]] = Field(default=None, description="Negative aspects of the movie")
    reviewer: Optional[str] = Field(default=None, description="Name of the reviewer")

structured_model = model.with_structured_output(MovieReview)

# Example movie review
result = structured_model.invoke("""I just watched 'Inception' directed by Christopher Nolan, and it was a mind-bending experience! Leonardo DiCaprio leads a stellar cast, and the concept of dreams within dreams is executed brilliantly. The visuals are stunning, and Hans Zimmer's soundtrack is unforgettable. Some parts of the plot are a bit confusing, but overall, it's a thrilling ride that keeps you thinking long after it's over.\n\nPros:\nAmazing visuals\nGreat acting\nUnique concept\nOutstanding soundtrack\n\nCons:\nComplex plot can be hard to follow\n\nReview by Sam Patel\nRating: 9.5/10\n""")

print(result) 