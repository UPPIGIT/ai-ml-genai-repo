"""
LangChain Output Parsing Examples
This file demonstrates various output parsing techniques for structured responses.
"""

import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import (
    PydanticOutputParser,
    ResponseSchema,
    StructuredOutputParser,
    CommaSeparatedListOutputParser,
    OutputFixingParser,
    RetryOutputParser
)
from pydantic import BaseModel, Field
from langchain.schema import OutputParserException

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def pydantic_output_parser_example():
    """Demonstrate Pydantic output parser for structured data."""
    print("=== Pydantic Output Parser Example ===")
    
    # Define the output structure using Pydantic
    class BookReview(BaseModel):
        title: str = Field(description="The title of the book")
        author: str = Field(description="The author of the book")
        rating: float = Field(description="Rating from 1-10")
        summary: str = Field(description="Brief summary of the book")
        strengths: List[str] = Field(description="Key strengths of the book")
        weaknesses: List[str] = Field(description="Areas for improvement")
        recommendation: str = Field(description="Who would enjoy this book")
        genre: str = Field(description="The genre of the book")
    
    # Create the parser
    parser = PydanticOutputParser(pydantic_object=BookReview)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Analyze the following book and provide a structured review:\n\n{book_description}\n\n{format_instructions}",
        input_variables=["book_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Test with a book description
    book_description = """
    "The Martian" by Andy Weir is a science fiction novel about an astronaut 
    who is stranded on Mars after his crew believes him dead and leaves him behind. 
    He must use his ingenuity and scientific knowledge to survive and find a way 
    to communicate with Earth.
    """
    
    formatted_prompt = prompt_template.format(book_description=book_description)
    print(f"Book Description: {book_description}")
    print(f"Prompt with Format Instructions:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = parser.parse(response.content)
        print(f"\nParsed Output:")
        print(f"Title: {parsed_output.title}")
        print(f"Author: {parsed_output.author}")
        print(f"Rating: {parsed_output.rating}/10")
        print(f"Genre: {parsed_output.genre}")
        print(f"Summary: {parsed_output.summary}")
        print(f"Strengths: {', '.join(parsed_output.strengths)}")
        print(f"Weaknesses: {', '.join(parsed_output.weaknesses)}")
        print(f"Recommendation: {parsed_output.recommendation}")
        
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Raw response: {response.content}\n")

def response_schema_parser_example():
    """Demonstrate ResponseSchema parser for flexible structured outputs."""
    print("=== Response Schema Parser Example ===")
    
    # Define response schemas
    response_schemas = [
        ResponseSchema(name="problem", description="The main problem identified"),
        ResponseSchema(name="root_cause", description="The root cause of the problem"),
        ResponseSchema(name="impact", description="The impact of the problem"),
        ResponseSchema(name="solutions", description="List of potential solutions"),
        ResponseSchema(name="priority", description="Priority level (High/Medium/Low)"),
        ResponseSchema(name="timeline", description="Estimated timeline to resolve")
    ]
    
    # Create the parser
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Analyze this technical issue:\n\n{issue_description}\n\n{format_instructions}",
        input_variables=["issue_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Test with a technical issue
    issue_description = """
    Our web application is experiencing intermittent 500 errors during peak hours.
    The errors seem to be related to database connection timeouts, and users are
    reporting slow response times and failed transactions.
    """
    
    formatted_prompt = prompt_template.format(issue_description=issue_description)
    print(f"Issue Description: {issue_description}")
    print(f"Prompt with Format Instructions:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = parser.parse(response.content)
        print(f"\nParsed Output:")
        for key, value in parsed_output.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Raw response: {response.content}\n")

def comma_separated_list_parser_example():
    """Demonstrate comma-separated list parser."""
    print("=== Comma-Separated List Parser Example ===")
    
    # Create the parser
    parser = CommaSeparatedListOutputParser()
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Generate a list of {topic} related to {domain}.\n\n{format_instructions}",
        input_variables=["topic", "domain"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Test with different topics
    test_cases = [
        ("programming languages", "web development"),
        ("machine learning algorithms", "data science"),
        ("cloud services", "AWS")
    ]
    
    for topic, domain in test_cases:
        formatted_prompt = prompt_template.format(topic=topic, domain=domain)
        print(f"Topic: {topic}")
        print(f"Domain: {domain}")
        print(f"Prompt: {formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        
        try:
            parsed_list = parser.parse(response.content)
            print(f"Parsed List: {parsed_list}")
            print(f"Number of items: {len(parsed_list)}\n")
            
        except Exception as e:
            print(f"Parsing error: {e}")
            print(f"Raw response: {response.content}\n")

def custom_parser_example():
    """Demonstrate custom output parser."""
    print("=== Custom Parser Example ===")
    
    class CodeReviewParser:
        """Custom parser for code review feedback."""
        
        def parse(self, text: str) -> Dict[str, any]:
            """Parse code review feedback into structured format."""
            
            # Define regex patterns for different sections
            patterns = {
                'issues': r'Issues?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                'suggestions': r'Suggestions?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                'improvements': r'Improvements?:\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                'rating': r'Rating:\s*(\d+)/10',
                'complexity': r'Complexity:\s*(Low|Medium|High)'
            }
            
            result = {}
            
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    if key == 'rating':
                        result[key] = int(match.group(1))
                    else:
                        result[key] = match.group(1).strip()
                else:
                    result[key] = None
            
            return result
        
        def get_format_instructions(self) -> str:
            """Get format instructions for the parser."""
            return """
            Please structure your code review response as follows:
            
            Issues: [List the main issues found]
            Suggestions: [Provide specific suggestions for improvement]
            Improvements: [Highlight positive aspects and improvements made]
            Rating: [Give a rating from 1-10]
            Complexity: [Rate the code complexity as Low, Medium, or High]
            """
    
    # Create the custom parser
    parser = CodeReviewParser()
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Review this code:\n\n{code}\n\n{format_instructions}",
        input_variables=["code"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Test with sample code
    sample_code = """
    def calculate_average(numbers):
        total = 0
        for num in numbers:
            total += num
        return total / len(numbers)
    """
    
    formatted_prompt = prompt_template.format(code=sample_code)
    print(f"Sample Code:\n{sample_code}")
    print(f"Prompt with Format Instructions:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = parser.parse(response.content)
        print(f"\nParsed Output:")
        for key, value in parsed_output.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Raw response: {response.content}\n")

def output_fixing_parser_example():
    """Demonstrate output fixing parser for handling parsing errors."""
    print("=== Output Fixing Parser Example ===")
    
    # Define the base parser
    class ProductReview(BaseModel):
        product_name: str = Field(description="Name of the product")
        rating: float = Field(description="Rating from 1-5")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")
        recommendation: bool = Field(description="Whether to recommend the product")
    
    base_parser = PydanticOutputParser(pydantic_object=ProductReview)
    
    # Create the fixing parser
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm
    )
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Review this product:\n\n{product_description}\n\n{format_instructions}",
        input_variables=["product_description"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )
    
    # Test with a product description
    product_description = """
    The iPhone 15 Pro features a titanium design, A17 Pro chip, and advanced camera system.
    It has excellent performance and build quality but is quite expensive.
    """
    
    formatted_prompt = prompt_template.format(product_description=product_description)
    print(f"Product Description: {product_description}")
    print(f"Prompt: {formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        # Try with the fixing parser
        parsed_output = fixing_parser.parse(response.content)
        print(f"\nParsed Output (Fixed):")
        print(f"Product Name: {parsed_output.product_name}")
        print(f"Rating: {parsed_output.rating}/5")
        print(f"Pros: {', '.join(parsed_output.pros)}")
        print(f"Cons: {', '.join(parsed_output.cons)}")
        print(f"Recommendation: {'Yes' if parsed_output.recommendation else 'No'}")
        
    except Exception as e:
        print(f"Parsing error even with fixing parser: {e}")
        print(f"Raw response: {response.content}\n")

def retry_parser_example():
    """Demonstrate retry parser for handling parsing failures."""
    print("=== Retry Parser Example ===")
    
    # Define a simple parser that might fail
    class SimpleParser:
        def parse(self, text: str) -> Dict[str, str]:
            """Parse text into a simple dictionary format."""
            # This parser expects a specific format and might fail
            if "Name:" not in text or "Age:" not in text:
                raise OutputParserException("Missing required fields")
            
            lines = text.strip().split('\n')
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip()] = value.strip()
            
            return result
        
        def get_format_instructions(self) -> str:
            return "Please provide the information in this format:\nName: [name]\nAge: [age]\nOccupation: [occupation]"
    
    # Create the retry parser
    base_parser = SimpleParser()
    retry_parser = RetryOutputParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=3
    )
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Extract information about this person:\n\n{person_description}\n\n{format_instructions}",
        input_variables=["person_description"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )
    
    # Test with a person description
    person_description = """
    John Smith is a 35-year-old software engineer who works at a tech company.
    He has 10 years of experience in Python development and loves hiking.
    """
    
    formatted_prompt = prompt_template.format(person_description=person_description)
    print(f"Person Description: {person_description}")
    print(f"Prompt: {formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = retry_parser.parse(response.content)
        print(f"\nParsed Output (with retries):")
        for key, value in parsed_output.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Parsing error after retries: {e}")
        print(f"Raw response: {response.content}\n")

def json_output_parser_example():
    """Demonstrate JSON output parsing."""
    print("=== JSON Output Parser Example ===")
    
    # Create a custom JSON parser
    class JSONParser:
        def parse(self, text: str) -> Dict:
            """Parse JSON from text response."""
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise OutputParserException("Invalid JSON format")
            else:
                raise OutputParserException("No JSON found in response")
        
        def get_format_instructions(self) -> str:
            return """
            Please respond with a valid JSON object in the following format:
            {
                "name": "string",
                "age": number,
                "skills": ["string"],
                "experience": "string",
                "rating": number
            }
            """
    
    # Create the parser
    parser = JSONParser()
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Analyze this developer profile:\n\n{profile}\n\n{format_instructions}",
        input_variables=["profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Test with a developer profile
    profile = """
    Sarah Johnson is a senior software engineer with 8 years of experience.
    She specializes in Python, JavaScript, and cloud technologies.
    Her colleagues rate her as a 9/10 developer with excellent problem-solving skills.
    """
    
    formatted_prompt = prompt_template.format(profile=profile)
    print(f"Developer Profile: {profile}")
    print(f"Prompt: {formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = parser.parse(response.content)
        print(f"\nParsed JSON Output:")
        print(json.dumps(parsed_output, indent=2))
        
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Raw response: {response.content}\n")

def main():
    """Run all output parsing examples."""
    print("LangChain Output Parsing Examples\n")
    print("=" * 50)
    
    try:
        pydantic_output_parser_example()
        response_schema_parser_example()
        comma_separated_list_parser_example()
        custom_parser_example()
        output_fixing_parser_example()
        retry_parser_example()
        json_output_parser_example()
        
        print("All output parsing examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 