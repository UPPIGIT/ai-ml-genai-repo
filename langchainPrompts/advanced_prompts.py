"""
Advanced LangChain Prompt Examples
This file demonstrates advanced prompt techniques including chain of thought, role-based prompts, and structured outputs.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def chain_of_thought_example():
    """Demonstrate chain of thought prompting for complex reasoning."""
    print("=== Chain of Thought Example ===")
    
    cot_prompt = """
    Let's approach this step by step:
    
    Problem: {problem}
    
    Let me think through this:
    1) First, I need to understand what's being asked
    2) Then, I'll break it down into smaller parts
    3) I'll solve each part systematically
    4) Finally, I'll combine the results
    
    Let's solve this step by step:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["problem"],
        template=cot_prompt
    )
    
    # Test with a complex problem
    problem = """
    A store sells apples for $2 each and oranges for $3 each. 
    If a customer buys 5 apples and 3 oranges, and pays with a $20 bill, 
    how much change should they receive?
    """
    
    formatted_prompt = prompt_template.format(problem=problem)
    print(f"Problem: {problem}")
    print(f"Chain of Thought Prompt:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}\n")

def role_based_prompt_example():
    """Demonstrate role-based prompting for different personas."""
    print("=== Role-Based Prompt Example ===")
    
    roles = {
        "expert": "You are a world-renowned expert in {field} with 20+ years of experience. Provide detailed, technical explanations with specific examples and best practices.",
        "teacher": "You are a patient and encouraging teacher explaining {field} to a beginner. Use simple language, analogies, and step-by-step explanations.",
        "consultant": "You are a strategic consultant advising on {field}. Focus on practical applications, ROI, and actionable recommendations.",
        "critic": "You are a critical analyst examining {field}. Consider potential issues, limitations, and alternative approaches."
    }
    
    field = "artificial intelligence"
    topic = "machine learning applications in healthcare"
    
    for role_name, role_description in roles.items():
        print(f"\n--- {role_name.upper()} PERSPECTIVE ---")
        
        prompt = f"""
        {role_description}
        
        Field: {field}
        Topic: {topic}
        
        Please provide your perspective on this topic.
        """
        
        response = llm.invoke(prompt)
        print(f"Response: {response.content[:300]}...\n")

def structured_output_example():
    """Demonstrate structured output parsing with Pydantic."""
    print("=== Structured Output Example ===")
    
    # Define the output structure
    class BookAnalysis(BaseModel):
        title: str = Field(description="The title of the book")
        author: str = Field(description="The author of the book")
        genre: str = Field(description="The genre of the book")
        summary: str = Field(description="A brief summary of the plot")
        themes: List[str] = Field(description="Key themes explored in the book")
        rating: float = Field(description="Rating from 1-10")
        recommendation: str = Field(description="Who would enjoy this book")
    
    # Create the parser
    parser = PydanticOutputParser(pydantic_object=BookAnalysis)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        template="Analyze the following book and provide a structured response:\n\n{book_description}\n\n{format_instructions}",
        input_variables=["book_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Test with a book description
    book_description = """
    "1984" by George Orwell is a dystopian novel set in a totalitarian society 
    where the government controls every aspect of people's lives through surveillance 
    and propaganda. The protagonist, Winston Smith, works at the Ministry of Truth 
    rewriting historical records to match the party's current version of events.
    """
    
    formatted_prompt = prompt_template.format(book_description=book_description)
    print(f"Book Description: {book_description}")
    print(f"Structured Prompt:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    
    try:
        parsed_output = parser.parse(response.content)
        print(f"Parsed Output:")
        print(f"Title: {parsed_output.title}")
        print(f"Author: {parsed_output.author}")
        print(f"Genre: {parsed_output.genre}")
        print(f"Summary: {parsed_output.summary}")
        print(f"Themes: {', '.join(parsed_output.themes)}")
        print(f"Rating: {parsed_output.rating}/10")
        print(f"Recommendation: {parsed_output.recommendation}\n")
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Raw response: {response.content}\n")

def multi_step_reasoning_example():
    """Demonstrate multi-step reasoning with intermediate steps."""
    print("=== Multi-Step Reasoning Example ===")
    
    reasoning_prompt = """
    You are a logical problem solver. For the given problem, break it down into clear steps and solve each step.
    
    Problem: {problem}
    
    Please structure your response as follows:
    
    STEP 1: [Identify what needs to be done]
    STEP 2: [Break down the problem]
    STEP 3: [Solve each part]
    STEP 4: [Combine results]
    STEP 5: [Verify the answer]
    
    FINAL ANSWER: [Your conclusion]
    """
    
    prompt_template = PromptTemplate(
        input_variables=["problem"],
        template=reasoning_prompt
    )
    
    # Test with a logic problem
    problem = """
    In a group of 100 people:
    - 60 people speak English
    - 40 people speak Spanish
    - 20 people speak both languages
    
    How many people speak neither English nor Spanish?
    """
    
    formatted_prompt = prompt_template.format(problem=problem)
    print(f"Problem: {problem}")
    print(f"Multi-Step Reasoning Prompt:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}\n")

def conditional_prompt_example():
    """Demonstrate conditional prompting based on input characteristics."""
    print("=== Conditional Prompt Example ===")
    
    def create_conditional_prompt(user_input, user_level):
        """Create different prompts based on user level."""
        
        if user_level == "beginner":
            template = """
            You are a patient teacher helping a complete beginner.
            
            User Question: {question}
            
            Please explain this in the simplest possible terms, using everyday analogies.
            Avoid technical jargon. If you must use technical terms, define them clearly.
            Provide step-by-step explanations.
            """
        elif user_level == "intermediate":
            template = """
            You are a knowledgeable instructor helping someone with some background knowledge.
            
            User Question: {question}
            
            Provide a balanced explanation with some technical details but still accessible.
            Include practical examples and common use cases.
            Mention related concepts that might be helpful to explore.
            """
        else:  # advanced
            template = """
            You are an expert discussing advanced concepts with a knowledgeable peer.
            
            User Question: {question}
            
            Provide a comprehensive, technical explanation with advanced concepts.
            Include implementation details, best practices, and potential challenges.
            Reference relevant research or industry standards where applicable.
            """
        
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )
    
    # Test with different user levels
    question = "What is machine learning?"
    user_levels = ["beginner", "intermediate", "advanced"]
    
    for level in user_levels:
        print(f"\n--- {level.upper()} LEVEL EXPLANATION ---")
        
        prompt_template = create_conditional_prompt(question, level)
        formatted_prompt = prompt_template.format(question=question)
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content[:400]}...\n")

def creative_prompt_example():
    """Demonstrate creative prompting techniques."""
    print("=== Creative Prompt Example ===")
    
    creative_prompts = {
        "metaphor": """
        Explain {concept} using a metaphor or analogy that would make sense to a {audience}.
        Make it memorable and engaging.
        """,
        
        "story": """
        Tell a short story that illustrates {concept}. 
        The story should be engaging and help the reader understand the concept naturally.
        """,
        
        "dialogue": """
        Create a dialogue between two characters discussing {concept}.
        One character should be knowledgeable, the other curious but confused.
        Make it natural and educational.
        """,
        
        "comparison": """
        Compare {concept} to something familiar from everyday life.
        Highlight the similarities and differences to make the concept clearer.
        """
    }
    
    concept = "blockchain technology"
    audience = "high school student"
    
    for style, template in creative_prompts.items():
        print(f"\n--- {style.upper()} STYLE ---")
        
        prompt_template = PromptTemplate(
            input_variables=["concept", "audience"],
            template=template
        )
        
        formatted_prompt = prompt_template.format(concept=concept, audience=audience)
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content[:300]}...\n")

def main():
    """Run all advanced prompt examples."""
    print("LangChain Advanced Prompt Examples\n")
    print("=" * 50)
    
    try:
        chain_of_thought_example()
        role_based_prompt_example()
        structured_output_example()
        multi_step_reasoning_example()
        conditional_prompt_example()
        creative_prompt_example()
        
        print("All advanced examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 