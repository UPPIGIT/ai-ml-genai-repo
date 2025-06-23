"""
Movie Review Output Parser Example (with Gemini and JsonOutputParser)
This example demonstrates how to use LangChain's JsonOutputParser to parse a movie review into a Python dict.
"""
from langchain.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt = PromptTemplate.from_template(
    """Extract the following fields from this movie review and return as a JSON object:
    - title
    - director
    - main_actors
    - key_themes
    - summary
    - rating
    - sentiment
    - pros
    - cons
    - reviewer

    Review: I just watched 'Inception' directed by Christopher Nolan, and it was a mind-bending experience! Leonardo DiCaprio leads a stellar cast, and the concept of dreams within dreams is executed brilliantly. The visuals are stunning, and Hans Zimmer's soundtrack is unforgettable. Some parts of the plot are a bit confusing, but overall, it's a thrilling ride that keeps you thinking long after it's over.

    Pros:
    Amazing visuals
    Great acting
    Unique concept
    Outstanding soundtrack

    Cons:
    Complex plot can be hard to follow

    Review by Sam Patel
    Rating: 9.5/10
    """
)

parser = JsonOutputParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed movie review (JSON):", result) 