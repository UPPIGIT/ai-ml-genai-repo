"""
LangChain Prompt Examples: From Basic to Advanced
===============================================

This file demonstrates various LangChain prompt patterns and techniques,
progressing from simple string templates to complex multi-step reasoning chains.
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import List

# =============================================================================
# BASIC EXAMPLES
# =============================================================================

def basic_prompt_template():
    """
    Basic string interpolation with PromptTemplate.
    Good for simple, single-variable substitutions.
    """
    # Simple template with one variable
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short poem about {topic}."
    )
    
    # Format the prompt
    formatted = prompt.format(topic="artificial intelligence")
    print("Basic Prompt:", formatted)
    return prompt

def multi_variable_prompt():
    """
    Template with multiple variables and more complex structure.
    Demonstrates how to organize prompts with multiple inputs.
    """
    prompt = PromptTemplate(
        input_variables=["language", "concept", "audience", "length"],
        template="""
        Explain the concept of {concept} in {language} programming.
        
        Audience: {audience}
        Length: {length}
        
        Please provide:
        1. A clear definition
        2. A practical example
        3. Common use cases
        """
    )
    
    formatted = prompt.format(
        language="Python",
        concept="decorators",
        audience="intermediate developers",
        length="2-3 paragraphs"
    )
    print("Multi-variable Prompt:", formatted)
    return prompt

# =============================================================================
# CHAT PROMPTS
# =============================================================================

def basic_chat_prompt():
    """
    Chat-based prompts for conversational AI models.
    Separates system instructions from user input for better control.
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding tutor who explains concepts clearly and provides practical examples."),
        ("human", "Explain {concept} in {language} with a simple example.")
    ])
    
    # Format for a specific query
    messages = chat_prompt.format_prompt(
        concept="list comprehensions",
        language="Python"
    ).to_messages()
    
    print("Chat Prompt Messages:")
    for msg in messages:
        print(f"{msg.__class__.__name__}: {msg.content}")
    
    return chat_prompt

def advanced_chat_prompt():
    """
    More sophisticated chat prompt with multiple message types.
    Includes system context, examples, and specific formatting instructions.
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert software architect with 15+ years of experience.
        You provide detailed, practical advice on system design and architecture patterns.
        Always include:
        - Trade-offs of your recommendations
        - Scalability considerations
        - Real-world examples when possible"""),
        
        ("human", "I need advice on designing a microservices architecture."),
        ("ai", "I'd be happy to help with microservices design. What specific aspects are you most concerned about - service boundaries, communication patterns, data management, or deployment strategies?"),
        ("human", "{user_question}")
    ])
    
    return chat_prompt

# =============================================================================
# FEW-SHOT PROMPTING
# =============================================================================

def few_shot_prompt_basic():
    """
    Few-shot learning with static examples.
    Helps the model understand the desired output format through examples.
    """
    # Define examples that show the pattern we want
    examples = [
        {
            "input": "Happy",
            "output": "I'm feeling joyful and content today, like sunshine breaking through clouds."
        },
        {
            "input": "Anxious", 
            "output": "My mind is racing with worries, like a storm brewing in my chest."
        },
        {
            "input": "Curious",
            "output": "I'm eager to explore and learn, like a child discovering a new playground."
        }
    ]
    
    # Template for each example
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Emotion: {input}\nDescription: {output}"
    )
    
    # Create few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Convert simple emotions into vivid, metaphorical descriptions:",
        suffix="Emotion: {emotion}\nDescription:",
        input_variables=["emotion"]
    )
    
    # Test with new emotion
    result = few_shot_prompt.format(emotion="Excited")
    print("Few-shot Basic Result:", result)
    return few_shot_prompt

def few_shot_with_selector():
    """
    Dynamic few-shot prompting with semantic similarity selection.
    Automatically selects the most relevant examples based on input similarity.
    """
    # More comprehensive examples for code review
    examples = [
        {
            "code": "def calculate_sum(numbers): return sum(numbers)",
            "review": "✅ Clean and concise. Consider adding type hints and docstring for better maintainability."
        },
        {
            "code": "for i in range(len(items)): print(items[i])",
            "review": "❌ Use 'for item in items:' instead of indexing. More Pythonic and readable."
        },
        {
            "code": "if user_input == 'yes' or user_input == 'y' or user_input == 'YES':",
            "review": "⚠️ Use 'user_input.lower() in ['yes', 'y']' for cleaner comparison logic."
        },
        {
            "code": "data = json.loads(response.text)",
            "review": "⚠️ Add error handling for JSON parsing. Consider using response.json() method."
        }
    ]
    
    # This would require actual embeddings in practice
    # Simplified version for demonstration
    example_prompt = PromptTemplate(
        input_variables=["code", "review"],
        template="Code: {code}\nReview: {review}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples[:2],  # Using subset for demo
        example_prompt=example_prompt,
        prefix="Provide constructive code reviews with specific suggestions:",
        suffix="Code: {code}\nReview:",
        input_variables=["code"]
    )
    
    return few_shot_prompt

# =============================================================================
# OUTPUT PARSING
# =============================================================================

class CodeReview(BaseModel):
    """Structured output model for code reviews."""
    rating: int = Field(description="Rating from 1-10")
    issues: List[str] = Field(description="List of identified issues")
    suggestions: List[str] = Field(description="List of improvement suggestions")
    positive_aspects: List[str] = Field(description="List of good practices found")

def structured_output_prompt():
    """
    Prompt that generates structured, parseable output.
    Uses Pydantic models to ensure consistent response format.
    """
    # Set up the output parser
    parser = PydanticOutputParser(pydantic_object=CodeReview)
    
    prompt = PromptTemplate(
        template="""
        You are an expert code reviewer. Analyze the following code and provide a structured review.
        
        Code to review:
        {code}
        
        {format_instructions}
        
        Focus on:
        - Code quality and readability
        - Best practices adherence
        - Potential bugs or security issues
        - Performance considerations
        """,
        input_variables=["code"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

# =============================================================================
# ADVANCED EXAMPLES
# =============================================================================

def chain_of_thought_prompt():
    """
    Chain-of-thought prompting for complex reasoning tasks.
    Encourages step-by-step thinking for better problem solving.
    """
    prompt = PromptTemplate(
        input_variables=["problem"],
        template="""
        Solve this step-by-step, showing your reasoning at each stage.
        
        Problem: {problem}
        
        Let's approach this systematically:
        
        Step 1: Understanding the problem
        - What are we trying to solve?
        - What information do we have?
        - What are the constraints?
        
        Step 2: Planning the solution
        - What approach should we take?
        - What are the key steps?
        - Are there any edge cases to consider?
        
        Step 3: Implementation
        - Work through the solution step by step
        - Show calculations or logic at each stage
        - Explain your reasoning
        
        Step 4: Verification
        - Does the solution make sense?
        - Can we verify our answer?
        - Are there alternative approaches?
        
        Begin your analysis:
        """
    )
    
    return prompt

def role_based_prompt():
    """
    Advanced role-based prompting with specific expertise and constraints.
    Creates detailed personas for specialized tasks.
    """
    prompt = PromptTemplate(
        input_variables=["technology", "scenario", "constraints"],
        template="""
        You are a Senior DevOps Engineer with 10+ years of experience in:
        - Cloud infrastructure (AWS, GCP, Azure)
        - Container orchestration (Kubernetes, Docker)
        - CI/CD pipelines and automation
        - Monitoring and observability
        - Security best practices
        
        Current context:
        - You're consulting for a mid-size tech company
        - They value pragmatic, cost-effective solutions
        - Security and reliability are top priorities
        - Team has mixed experience levels
        
        Task: Design a solution for {scenario} using {technology}
        
        Constraints:
        {constraints}
        
        Please provide:
        1. High-level architecture overview
        2. Step-by-step implementation plan
        3. Risk assessment and mitigation strategies
        4. Cost considerations and optimizations
        5. Monitoring and maintenance recommendations
        
        Format your response as a technical proposal that could be presented to stakeholders.
        """
    )
    
    return prompt

def multi_step_reasoning_prompt():
    """
    Complex prompt for multi-step reasoning and analysis.
    Combines multiple AI capabilities in a structured workflow.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI research assistant capable of multi-step analysis.
        For each query, you will:
        1. Break down the problem into components
        2. Analyze each component systematically  
        3. Synthesize findings into actionable insights
        4. Provide confidence levels for your conclusions"""),
        
        ("human", """
        Research Topic: {topic}
        
        Please conduct a comprehensive analysis following this structure:
        
        PHASE 1: Problem Decomposition
        - Identify 3-5 key aspects of this topic
        - Explain why each aspect is important
        - Note any interdependencies
        
        PHASE 2: Component Analysis
        For each aspect identified:
        - Current state of knowledge
        - Key challenges or unknowns
        - Recent developments or trends
        - Confidence level (High/Medium/Low)
        
        PHASE 3: Synthesis
        - How do the components relate to each other?
        - What are the most critical findings?
        - What questions remain unanswered?
        
        PHASE 4: Actionable Insights
        - Top 3 recommendations
        - Next steps for further research
        - Potential implications or applications
        
        Begin your analysis:
        """)
    ])
    
    return prompt

def conditional_prompt():
    """
    Advanced prompt with conditional logic and dynamic content.
    Adapts behavior based on input characteristics.
    """
    prompt = PromptTemplate(
        input_variables=["user_type", "experience_level", "question", "context"],
        template="""
        {{% if user_type == "student" %}}
        You are a patient teacher who explains concepts clearly with examples.
        Adjust your explanation for {experience_level} level understanding.
        {{% elif user_type == "professional" %}}
        You are a knowledgeable colleague providing practical, actionable advice.
        Assume {experience_level} level expertise and focus on implementation details.
        {{% else %}}
        You are a helpful assistant providing balanced information for general audience.
        {{% endif %}}
        
        Context: {context}
        
        Question: {question}
        
        {{% if experience_level == "beginner" %}}
        Please include:
        - Clear definitions of technical terms
        - Step-by-step explanations
        - Simple examples or analogies
        {{% elif experience_level == "intermediate" %}}
        Please include:
        - Practical examples and use cases
        - Best practices and common pitfalls
        - References to related concepts
        {{% else %}}
        Please include:
        - Advanced considerations and edge cases
        - Performance and scalability implications
        - Integration with other systems/concepts
        {{% endif %}}
        """)
    
    return prompt

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def demonstrate_prompts():
    """
    Function to demonstrate various prompt types.
    Run this to see all examples in action.
    """
    print("=" * 60)
    print("LANGCHAIN PROMPT EXAMPLES DEMONSTRATION")
    print("=" * 60)
    
    # Basic examples
    print("\n1. Basic Prompt Template:")
    basic_prompt_template()
    
    print("\n2. Multi-variable Prompt:")
    multi_variable_prompt()
    
    # Chat examples
    print("\n3. Basic Chat Prompt:")
    basic_chat_prompt()
    
    # Few-shot examples
    print("\n4. Few-shot Prompt:")
    few_shot_prompt_basic()
    
    # Structured output
    print("\n5. Structured Output Prompt:")
    structured_prompt, parser = structured_output_prompt()
    sample_code = "def process_data(data): return [x*2 for x in data if x > 0]"
    formatted = structured_prompt.format(code=sample_code)
    print("Formatted prompt preview:", formatted[:200] + "...")
    
    print("\n" + "=" * 60)
    print("All prompt examples demonstrated!")
    print("=" * 60)

# Example usage patterns for different scenarios
PROMPT_PATTERNS = {
    "data_analysis": """
        Analyze the following dataset: {data}
        
        Please provide:
        1. Summary statistics
        2. Key insights and patterns
        3. Potential data quality issues
        4. Recommendations for further analysis
    """,
    
    "code_generation": """
        Generate {language} code for: {requirement}
        
        Requirements:
        - Follow best practices and coding standards
        - Include error handling
        - Add comprehensive comments
        - Provide usage examples
        
        Code:
    """,
    
    "document_summarization": """
        Summarize the following document in {style} style for {audience}:
        
        Document: {document}
        
        Summary length: {length}
        Key focus areas: {focus_areas}
        
        Summary:
    """
}

if __name__ == "__main__":
    # Run demonstration
    demonstrate_prompts()
