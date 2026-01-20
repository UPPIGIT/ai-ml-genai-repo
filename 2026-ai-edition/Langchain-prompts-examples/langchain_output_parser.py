# 07_prompts_with_output_parsers.py
# Output parsers work with prompts to structure and validate AI responses
# They ensure you get data in a specific format (JSON, lists, etc.)

from langchain.prompts import PromptTemplate
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    StructuredOutputParser,
    ResponseSchema,
    PydanticOutputParser,
    DatetimeOutputParser,
)
from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime

# Example 1: Comma-Separated List Output Parser
# Ensures the AI returns a simple list of items
list_parser = CommaSeparatedListOutputParser()

# Get format instructions to tell AI how to respond
format_instructions = list_parser.get_format_instructions()
print("Example 1 - List Parser Instructions:")
print(format_instructions)
print()

list_prompt = PromptTemplate(
    template="List 5 {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": format_instructions}
)

prompt1 = list_prompt.format(topic="programming languages")
print("Generated Prompt:")
print(prompt1)

# Simulated AI response
ai_response1 = "Python, JavaScript, Java, C++, Ruby"
parsed_output1 = list_parser.parse(ai_response1)
print("\nParsed Output:")
print(parsed_output1)  # Returns a Python list
print(f"Type: {type(parsed_output1)}")
print("\n" + "="*50 + "\n")

# Example 2: Structured Output Parser with Response Schemas
# Define exactly what fields you want in the response
response_schemas = [
    ResponseSchema(
        name="name",
        description="The name of the person"
    ),
    ResponseSchema(
        name="age",
        description="The age of the person in years"
    ),
    ResponseSchema(
        name="occupation",
        description="The person's job or profession"
    ),
    ResponseSchema(
        name="location",
        description="Where the person lives"
    )
]

structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions2 = structured_parser.get_format_instructions()

structured_prompt = PromptTemplate(
    template="Extract information about the person from this text: {text}\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions2}
)

prompt2 = structured_prompt.format(
    text="John Smith is a 35-year-old software engineer living in San Francisco."
)
print("Example 2 - Structured Parser:")
print(prompt2)

# Simulated AI response
ai_response2 = """```json
{
    "name": "John Smith",
    "age": "35",
    "occupation": "software engineer",
    "location": "San Francisco"
}
```"""
parsed_output2 = structured_parser.parse(ai_response2)
print("\nParsed Output:")
print(parsed_output2)  # Returns a dictionary
print(f"Name: {parsed_output2['name']}, Age: {parsed_output2['age']}")
print("\n" + "="*50 + "\n")

# Example 3: Pydantic Output Parser (Type-Safe)
# Uses Pydantic models for strong typing and validation
class Book(BaseModel):
    """Information about a book"""
    title: str = Field(description="The title of the book")
    author: str = Field(description="The author's name")
    year: int = Field(description="Year of publication")
    genre: str = Field(description="The genre of the book")
    pages: int = Field(description="Number of pages")
    
    @field_validator('year')
    @classmethod
    def year_must_be_valid(cls, v):
        if v < 1000 or v > datetime.now().year:
            raise ValueError('Year must be between 1000 and current year')
        return v

pydantic_parser = PydanticOutputParser(pydantic_object=Book)
format_instructions3 = pydantic_parser.get_format_instructions()

pydantic_prompt = PromptTemplate(
    template="Extract book information: {query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions3}
)

prompt3 = pydantic_prompt.format(
    query="Tell me about '1984' by George Orwell, published in 1949, a dystopian fiction novel with 328 pages."
)
print("Example 3 - Pydantic Parser:")
print(prompt3[:200] + "...")  # Truncated for readability

# Simulated AI response
ai_response3 = """```json
{
    "title": "1984",
    "author": "George Orwell",
    "year": 1949,
    "genre": "dystopian fiction",
    "pages": 328
}
```"""
parsed_output3 = pydantic_parser.parse(ai_response3)
print("\nParsed Output:")
print(parsed_output3)
print(f"Type: {type(parsed_output3)}")  # It's a Book object!
print(f"Title: {parsed_output3.title}, Pages: {parsed_output3.pages}")
print("\n" + "="*50 + "\n")

# Example 4: Complex Nested Structure
# Parse complex nested data structures
class Address(BaseModel):
    street: str
    city: str
    country: str
    zipcode: str

class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    address: Address = Field(description="Person's address")
    skills: List[str] = Field(description="List of skills")
    experience_years: int = Field(description="Years of experience")

complex_parser = PydanticOutputParser(pydantic_object=Person)
format_instructions4 = complex_parser.get_format_instructions()

complex_prompt = PromptTemplate(
    template="Create a professional profile: {description}\n{format_instructions}",
    input_variables=["description"],
    partial_variables={"format_instructions": format_instructions4}
)

prompt4 = complex_prompt.format(
    description="Sarah Johnson, email sarah@email.com, lives at 123 Main St, Boston, USA, 02101. She has 8 years experience and knows Python, SQL, and Machine Learning."
)
print("Example 4 - Complex Nested Structure:")
print(prompt4[:250] + "...")

# Simulated AI response
ai_response4 = """```json
{
    "name": "Sarah Johnson",
    "email": "sarah@email.com",
    "address": {
        "street": "123 Main St",
        "city": "Boston",
        "country": "USA",
        "zipcode": "02101"
    },
    "skills": ["Python", "SQL", "Machine Learning"],
    "experience_years": 8
}
```"""
parsed_output4 = complex_parser.parse(ai_response4)
print("\nParsed Output:")
print(f"Name: {parsed_output4.name}")
print(f"City: {parsed_output4.address.city}")
print(f"Skills: {', '.join(parsed_output4.skills)}")
print("\n" + "="*50 + "\n")

# Example 5: Custom Output Parser
# Create your own parser for special formats
from langchain.schema import BaseOutputParser

class BulletPointParser(BaseOutputParser):
    """Parse output into bullet points"""
    
    def parse(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        lines = text.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            # Remove common bullet point markers
            cleaned = line.strip()
            for marker in ['- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                if cleaned.startswith(marker):
                    cleaned = cleaned[len(marker):]
                    break
            
            if cleaned:  # Only add non-empty lines
                bullet_points.append(cleaned)
        
        return bullet_points
    
    def get_format_instructions(self) -> str:
        return "Provide your answer as a bulleted list with each point on a new line, starting with '- '"

bullet_parser = BulletPointParser()
format_instructions5 = bullet_parser.get_format_instructions()

bullet_prompt = PromptTemplate(
    template="What are the benefits of {topic}?\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": format_instructions5}
)

prompt5 = bullet_prompt.format(topic="exercise")
print("Example 5 - Custom Bullet Point Parser:")
print(prompt5)

# Simulated AI response
ai_response5 = """- Improves cardiovascular health
- Increases muscle strength
- Boosts mental well-being
- Helps with weight management
- Enhances sleep quality"""

parsed_output5 = bullet_parser.parse(ai_response5)
print("\nParsed Output:")
for i, point in enumerate(parsed_output5, 1):
    print(f"{i}. {point}")
print(f"\nTotal benefits listed: {len(parsed_output5)}")