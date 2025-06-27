"""
LangChain Prompts with Output Parsers: Complete Examples
=======================================================

This file demonstrates how to combine LangChain prompts with various output parsers
to get structured, reliable outputs from language models.
"""

import json
import re
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser,
    RetryOutputParser,
    StructuredOutputParser,
    ResponseSchema,
    CommaSeparatedListOutputParser,
    DatetimeOutputParser,
    EnumOutputParser
)
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field, validator
from langchain.llms.base import BaseLLM

# =============================================================================
# PYDANTIC OUTPUT PARSERS
# =============================================================================

class TaskPriority(str, Enum):
    """Enum for task priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Task(BaseModel):
    """Model for individual tasks."""
    title: str = Field(description="Brief title of the task")
    description: str = Field(description="Detailed description of what needs to be done")
    priority: TaskPriority = Field(description="Priority level of the task")
    estimated_hours: float = Field(description="Estimated hours to complete", ge=0.1, le=40.0)
    tags: List[str] = Field(description="List of relevant tags or categories")
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

class ProjectPlan(BaseModel):
    """Model for complete project planning."""
    project_name: str = Field(description="Name of the project")
    description: str = Field(description="Project overview and objectives")
    tasks: List[Task] = Field(description="List of tasks to complete the project")
    total_estimated_hours: float = Field(description="Sum of all task hours")
    completion_deadline: str = Field(description="Target completion date in YYYY-MM-DD format")
    
    @validator('total_estimated_hours')
    def validate_total_hours(cls, v, values):
        if 'tasks' in values:
            calculated_total = sum(task.estimated_hours for task in values['tasks'])
            if abs(v - calculated_total) > 0.1:  # Allow small rounding differences
                raise ValueError(f'Total hours {v} does not match sum of task hours {calculated_total}')
        return v

def pydantic_project_planning_example():
    """
    Complex Pydantic parser example for project planning.
    Demonstrates nested models, validation, and enums.
    """
    # Set up the parser
    parser = PydanticOutputParser(pydantic_object=ProjectPlan)
    
    prompt = PromptTemplate(
        template="""
        You are a project manager creating a detailed project plan.
        
        Project Request: {project_request}
        
        Create a comprehensive project plan with the following requirements:
        - Break down the project into 3-7 specific, actionable tasks
        - Assign realistic priority levels and time estimates
        - Include relevant tags for each task
        - Calculate total project time
        - Set a realistic completion deadline
        
        {format_instructions}
        
        Important: Ensure all tasks are specific, measurable, and achievable.
        """,
        input_variables=["project_request"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Example usage
    sample_request = "Build a REST API for a task management application with user authentication"
    formatted_prompt = prompt.format(project_request=sample_request)
    
    print("=== PYDANTIC PROJECT PLANNING EXAMPLE ===")
    print("Prompt:")
    print(formatted_prompt)
    print("\nParser format instructions:")
    print(parser.get_format_instructions())
    
    return prompt, parser

# =============================================================================
# STRUCTURED OUTPUT PARSERS
# =============================================================================

def structured_code_review_example():
    """
    Structured output parser for code reviews.
    Uses ResponseSchema for flexible structured output without Pydantic.
    """
    # Define the response structure
    response_schemas = [
        ResponseSchema(
            name="overall_rating",
            description="Overall code quality rating from 1-10"
        ),
        ResponseSchema(
            name="code_style_score",
            description="Code style and formatting score from 1-10"
        ),
        ResponseSchema(
            name="performance_score", 
            description="Performance and efficiency score from 1-10"
        ),
        ResponseSchema(
            name="security_score",
            description="Security considerations score from 1-10"
        ),
        ResponseSchema(
            name="critical_issues",
            description="List of critical issues that must be fixed"
        ),
        ResponseSchema(
            name="suggestions",
            description="List of improvement suggestions"
        ),
        ResponseSchema(
            name="positive_aspects",
            description="List of things done well in the code"
        ),
        ResponseSchema(
            name="refactoring_priority",
            description="Priority level for refactoring: low, medium, high, or critical"
        )
    ]
    
    # Create the parser
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    prompt = PromptTemplate(
        template="""
        You are a senior software engineer conducting a thorough code review.
        
        Code to review:
        ```{language}
        {code}
        ```
        
        Context: {context}
        
        Please provide a comprehensive code review covering:
        - Code quality and readability
        - Performance implications
        - Security considerations
        - Best practices adherence
        - Potential bugs or issues
        
        Be constructive and specific in your feedback.
        
        {format_instructions}
        """,
        input_variables=["code", "language", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print("\n=== STRUCTURED CODE REVIEW EXAMPLE ===")
    print("Parser format instructions:")
    print(parser.get_format_instructions())
    
    return prompt, parser

# =============================================================================
# LIST OUTPUT PARSERS
# =============================================================================

def comma_separated_list_example():
    """
    Simple comma-separated list parser.
    Useful for generating lists of items, keywords, etc.
    """
    parser = CommaSeparatedListOutputParser()
    
    prompt = PromptTemplate(
        template="""
        Generate a comprehensive list of {item_type} for {context}.
        
        Requirements:
        - Include {count} items
        - Focus on {focus_area}
        - Items should be {criteria}
        
        {format_instructions}
        
        List:
        """,
        input_variables=["item_type", "context", "count", "focus_area", "criteria"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print("\n=== COMMA SEPARATED LIST EXAMPLE ===")
    print("Parser format instructions:")
    print(parser.get_format_instructions())
    
    return prompt, parser

# =============================================================================
# CUSTOM OUTPUT PARSERS
# =============================================================================

class JSONOutputParser:
    """
    Custom parser for JSON output with error handling.
    Demonstrates creating custom parsers for specific needs.
    """
    
    def get_format_instructions(self) -> str:
        return """
        Your response must be a valid JSON object with the following structure:
        {
            "response_type": "string indicating the type of response",
            "data": "the main content of your response",
            "metadata": {
                "confidence": "number between 0 and 1",
                "sources": ["list of sources if applicable"],
                "timestamp": "current timestamp"
            }
        }
        
        Ensure the JSON is properly formatted and contains all required fields.
        """
    
    def parse(self, text: str) -> Dict:
        """Parse the output text into a structured format."""
        try:
            # Extract JSON from the text (in case there's additional text)
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = text.strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["response_type", "data", "metadata"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate metadata structure
            metadata = parsed["metadata"]
            if "confidence" not in metadata:
                raise ValueError("Missing confidence in metadata")
            
            confidence = metadata["confidence"]
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                raise ValueError("Confidence must be a number between 0 and 1")
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise OutputParserException(f"Invalid JSON format: {e}")
        except Exception as e:
            raise OutputParserException(f"Parsing error: {e}")

def custom_json_parser_example():
    """
    Example using custom JSON parser for structured responses.
    """
    parser = JSONOutputParser()
    
    prompt = PromptTemplate(
        template="""
        You are a research assistant providing detailed analysis.
        
        Query: {query}
        Context: {context}
        
        Please analyze the query and provide a comprehensive response.
        Include your confidence level and any relevant sources.
        
        {format_instructions}
        """,
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print("\n=== CUSTOM JSON PARSER EXAMPLE ===")
    print("Parser format instructions:")
    print(parser.get_format_instructions())
    
    return prompt, parser

# =============================================================================
# ERROR HANDLING WITH OUTPUT FIXING PARSER
# =============================================================================

class RobustTaskParser(BaseModel):
    """Task model with comprehensive validation."""
    task_id: str = Field(description="Unique identifier for the task")
    title: str = Field(description="Task title")
    status: str = Field(description="Task status: todo, in_progress, done, or blocked")
    assignee: Optional[str] = Field(description="Person assigned to the task")
    due_date: Optional[str] = Field(description="Due date in YYYY-MM-DD format")
    labels: List[str] = Field(description="List of labels/tags")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['todo', 'in_progress', 'done', 'blocked']
        if v.lower() not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v.lower()
    
    @validator('due_date')
    def validate_due_date(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Due date must be in YYYY-MM-DD format')

def output_fixing_parser_example():
    """
    Example using OutputFixingParser to handle malformed outputs.
    This parser attempts to fix common formatting issues automatically.
    """
    # Base parser
    base_parser = PydanticOutputParser(pydantic_object=RobustTaskParser)
    
    # This would typically wrap the base parser with an LLM for fixing
    # For demonstration, we'll show the concept
    prompt = PromptTemplate(
        template="""
        Create a task from the following description: {task_description}
        
        Generate a unique task ID, determine appropriate status, and extract relevant information.
        If no due date is mentioned, leave it as null.
        
        {format_instructions}
        
        Task details:
        """,
        input_variables=["task_description"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )
    
    print("\n=== OUTPUT FIXING PARSER EXAMPLE ===")
    print("This parser can automatically fix common JSON formatting issues.")
    print("Base parser format instructions:")
    print(base_parser.get_format_instructions())
    
    return prompt, base_parser

# =============================================================================
# DATETIME OUTPUT PARSER
# =============================================================================

def datetime_parser_example():
    """
    Example using DatetimeOutputParser for handling date/time extraction.
    """
    parser = DatetimeOutputParser()
    
    prompt = PromptTemplate(
        template="""
        Extract the most relevant date/time from the following text: {text}
        
        If multiple dates are mentioned, choose the most important one (usually the deadline or event date).
        If no specific date is mentioned, make a reasonable assumption based on context.
        
        Text: {text}
        Context: {context}
        
        {format_instructions}
        """,
        input_variables=["text", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print("\n=== DATETIME PARSER EXAMPLE ===")
    print("Parser format instructions:")
    print(parser.get_format_instructions())
    
    return prompt, parser

# =============================================================================
# ENUM OUTPUT PARSER
# =============================================================================

class SentimentType(Enum):
    """Sentiment analysis enumeration."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

def enum_parser_example():
    """
    Example using EnumOutputParser for controlled vocabulary responses.
    """
    parser = EnumOutputParser(enum=SentimentType)
    
    prompt = PromptTemplate(
        template="""
        Analyze the sentiment of the following text: {text}
        
        Consider:
        - Overall emotional tone
        - Language intensity
        - Context and implications
        
        Text: "{text}"
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print("\n=== ENUM PARSER EXAMPLE ===")
    print("Parser format instructions:")
    print(parser.get_format_instructions())
    
    return prompt, parser

# =============================================================================
# COMPLEX MULTI-STEP PARSING
# =============================================================================

class AnalysisStep(BaseModel):
    """Individual analysis step."""
    step_number: int = Field(description="Step number in the analysis")
    step_name: str = Field(description="Name of this analysis step")
    findings: List[str] = Field(description="Key findings from this step")
    confidence: float = Field(description="Confidence level 0-1", ge=0, le=1)
    next_actions: List[str] = Field(description="Recommended next actions")

class ComprehensiveAnalysis(BaseModel):
    """Complete multi-step analysis result."""
    analysis_id: str = Field(description="Unique identifier for this analysis")
    topic: str = Field(description="Topic being analyzed")
    methodology: str = Field(description="Analysis methodology used")
    steps: List[AnalysisStep] = Field(description="Individual analysis steps")
    overall_conclusion: str = Field(description="Overall conclusion from the analysis")
    recommendations: List[str] = Field(description="Final recommendations")
    limitations: List[str] = Field(description="Analysis limitations and caveats")
    
    @validator('steps')
    def validate_steps(cls, v):
        if len(v) < 2:
            raise ValueError('Analysis must have at least 2 steps')
        return v

def multi_step_analysis_parser():
    """
    Complex parser for multi-step analysis with validation.
    Demonstrates handling complex nested structures.
    """
    parser = PydanticOutputParser(pydantic_object=ComprehensiveAnalysis)
    
    prompt = PromptTemplate(
        template="""
        You are a research analyst conducting a comprehensive multi-step analysis.
        
        Analysis Request: {request}
        Available Data: {data}
        Analysis Type: {analysis_type}
        
        Conduct a thorough analysis following these steps:
        1. Problem identification and scoping
        2. Data examination and quality assessment
        3. Primary analysis and pattern identification
        4. Secondary analysis and validation
        5. Conclusion synthesis and recommendation development
        
        For each step:
        - Document key findings
        - Assess confidence level
        - Identify next actions
        
        Provide an overall methodology description and final recommendations.
        Be honest about limitations and uncertainties.
        
        {format_instructions}
        """,
        input_variables=["request", "data", "analysis_type"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print("\n=== MULTI-STEP ANALYSIS PARSER ===")
    print("This parser handles complex nested analysis structures.")
    
    return prompt, parser

# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def demonstrate_all_parsers():
    """
    Demonstrate all parser types with sample inputs.
    """
    print("=" * 80)
    print("LANGCHAIN PROMPTS WITH OUTPUT PARSERS - COMPLETE EXAMPLES")
    print("=" * 80)
    
    # Pydantic parser
    proj_prompt, proj_parser = pydantic_project_planning_example()
    
    # Structured parser
    review_prompt, review_parser = structured_code_review_example()
    
    # List parser
    list_prompt, list_parser = comma_separated_list_example()
    
    # Custom JSON parser
    json_prompt, json_parser = custom_json_parser_example()
    
    # Output fixing parser
    fixing_prompt, fixing_parser = output_fixing_parser_example()
    
    # Datetime parser
    dt_prompt, dt_parser = datetime_parser_example()
    
    # Enum parser
    enum_prompt, enum_parser = enum_parser_example()
    
    # Multi-step analysis
    analysis_prompt, analysis_parser = multi_step_analysis_parser()
    
    print("\n" + "=" * 80)
    print("ALL PARSER EXAMPLES DEMONSTRATED!")
    print("=" * 80)
    
    return {
        'project_planning': (proj_prompt, proj_parser),
        'code_review': (review_prompt, review_parser),
        'list_generation': (list_prompt, list_parser),
        'json_response': (json_prompt, json_parser),
        'error_fixing': (fixing_prompt, fixing_parser),
        'datetime_extraction': (dt_prompt, dt_parser),
        'sentiment_analysis': (enum_prompt, enum_parser),
        'multi_step_analysis': (analysis_prompt, analysis_parser)
    }

# Example usage with mock LLM responses
SAMPLE_RESPONSES = {
    'project_planning': """
    {
        "project_name": "Task Management API",
        "description": "Build a REST API for task management with user authentication",
        "tasks": [
            {
                "title": "Set up project structure",
                "description": "Initialize project with proper folder structure and dependencies",
                "priority": "high",
                "estimated_hours": 4.0,
                "tags": ["setup", "architecture"]
            },
            {
                "title": "Implement user authentication",
                "description": "Create JWT-based authentication system with login/register endpoints",
                "priority": "high", 
                "estimated_hours": 8.0,
                "tags": ["auth", "security", "jwt"]
            },
            {
                "title": "Build task CRUD operations",
                "description": "Create endpoints for creating, reading, updating, and deleting tasks",
                "priority": "medium",
                "estimated_hours": 12.0,
                "tags": ["crud", "api", "database"]
            }
        ],
        "total_estimated_hours": 24.0,
        "completion_deadline": "2025-07-15"
    }
    """,
    
    'sentiment_analysis': "positive"
}

def test_parsers_with_samples():
    """
    Test parsers with sample responses to show parsing in action.
    """
    print("\n" + "=" * 60)
    print("TESTING PARSERS WITH SAMPLE RESPONSES")
    print("=" * 60)
    
    parsers = demonstrate_all_parsers()
    
    # Test project planning parser
    print("\n--- Testing Project Planning Parser ---")
    try:
        _, proj_parser = parsers['project_planning']
        result = proj_parser.parse(SAMPLE_RESPONSES['project_planning'])
        print("✅ Successfully parsed project plan:")
        print(f"Project: {result.project_name}")
        print(f"Tasks: {len(result.tasks)}")
        print(f"Total hours: {result.total_estimated_hours}")
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
    
    # Test sentiment parser
    print("\n--- Testing Sentiment Parser ---")
    try:
        _, sentiment_parser = parsers['sentiment_analysis']
        result = sentiment_parser.parse(SAMPLE_RESPONSES['sentiment_analysis'])
        print(f"✅ Successfully parsed sentiment: {result}")
    except Exception as e:
        print(f"❌ Parsing failed: {e}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_all_parsers()
    test_parsers_with_samples()
