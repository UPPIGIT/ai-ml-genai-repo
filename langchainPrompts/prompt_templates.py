"""
LangChain Prompt Templates
This file demonstrates reusable prompt templates and dynamic prompt generation.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.loading import load_prompt_from_config
import json

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def reusable_templates_example():
    """Demonstrate reusable prompt templates for common tasks."""
    print("=== Reusable Templates Example ===")
    
    # Define reusable templates
    templates = {
        "code_review": PromptTemplate(
            input_variables=["language", "code", "focus_areas"],
            template="""
            You are an expert {language} code reviewer.
            
            Please review the following code:
            {code}
            
            Focus on: {focus_areas}
            
            Provide feedback on:
            1. Code quality and best practices
            2. Potential bugs or issues
            3. Performance considerations
            4. Security concerns
            5. Suggestions for improvement
            """
        ),
        
        "documentation": PromptTemplate(
            input_variables=["function_name", "parameters", "return_type", "description"],
            template="""
            Write comprehensive documentation for the following function:
            
            Function: {function_name}
            Parameters: {parameters}
            Return Type: {return_type}
            Description: {description}
            
            Include:
            - Function purpose
            - Parameter descriptions
            - Return value explanation
            - Usage examples
            - Edge cases or limitations
            """
        ),
        
        "error_analysis": PromptTemplate(
            input_variables=["error_message", "code_context", "language"],
            template="""
            Analyze this {language} error:
            
            Error: {error_message}
            Code Context: {code_context}
            
            Please provide:
            1. What caused this error
            2. How to fix it
            3. How to prevent it in the future
            4. Alternative approaches
            """
        )
    }
    
    # Test code review template
    print("--- Code Review Template ---")
    code_review_prompt = templates["code_review"].format(
        language="Python",
        code="""
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
        """,
        focus_areas="error handling, edge cases, code efficiency"
    )
    
    print(f"Code Review Prompt:\n{code_review_prompt}")
    response = llm.invoke(code_review_prompt)
    print(f"Response: {response.content[:300]}...\n")
    
    # Test documentation template
    print("--- Documentation Template ---")
    doc_prompt = templates["documentation"].format(
        function_name="calculate_fibonacci",
        parameters="n: int",
        return_type="int",
        description="Calculate the nth Fibonacci number"
    )
    
    print(f"Documentation Prompt:\n{doc_prompt}")
    response = llm.invoke(doc_prompt)
    print(f"Response: {response.content[:300]}...\n")

def pipeline_prompt_example():
    """Demonstrate pipeline prompt templates for complex workflows."""
    print("=== Pipeline Prompt Example ===")
    
    # Define the base template
    base_template = """
    You are a {role} with expertise in {domain}.
    
    {task_description}
    
    Context: {context}
    """
    
    # Define the refinement template
    refinement_template = """
    {base_response}
    
    Now, please refine this response to be more {tone} and suitable for {audience}.
    Focus on {focus_areas}.
    """
    
    # Create the pipeline
    base_prompt = PromptTemplate(
        input_variables=["role", "domain", "task_description", "context"],
        template=base_template
    )
    
    refinement_prompt = PromptTemplate(
        input_variables=["base_response", "tone", "audience", "focus_areas"],
        template=refinement_template
    )
    
    # Create the pipeline prompt
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=refinement_prompt,
        pipeline_prompts=[("base_response", base_prompt)]
    )
    
    # Test the pipeline
    pipeline_inputs = {
        "role": "data scientist",
        "domain": "machine learning",
        "task_description": "Explain the concept of overfitting in machine learning",
        "context": "This is for a technical blog post",
        "tone": "engaging and accessible",
        "audience": "software developers new to ML",
        "focus_areas": "practical implications and prevention strategies"
    }
    
    formatted_prompt = pipeline_prompt.format(**pipeline_inputs)
    print(f"Pipeline Prompt:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content[:400]}...\n")

def dynamic_template_generation():
    """Demonstrate dynamic template generation based on context."""
    print("=== Dynamic Template Generation ===")
    
    def generate_template_for_task(task_type, complexity, audience):
        """Generate a template based on task characteristics."""
        
        base_templates = {
            "explanation": {
                "simple": "Explain {topic} in simple terms that a {audience} can understand.",
                "detailed": "Provide a comprehensive explanation of {topic} including background, key concepts, and practical applications.",
                "technical": "Give a technical deep-dive into {topic} with implementation details and advanced concepts."
            },
            "analysis": {
                "simple": "Analyze {topic} and provide key insights in simple terms.",
                "detailed": "Conduct a thorough analysis of {topic} with supporting evidence and multiple perspectives.",
                "technical": "Perform a technical analysis of {topic} with quantitative metrics and detailed methodology."
            },
            "comparison": {
                "simple": "Compare {topic1} and {topic2} in simple terms.",
                "detailed": "Provide a detailed comparison of {topic1} and {topic2} across multiple dimensions.",
                "technical": "Conduct a technical comparison of {topic1} and {topic2} with performance metrics and benchmarks."
            }
        }
        
        template_text = base_templates[task_type][complexity]
        
        # Add audience-specific instructions
        audience_instructions = {
            "beginner": "Use simple language and provide examples.",
            "intermediate": "Balance technical detail with accessibility.",
            "expert": "Focus on advanced concepts and technical depth."
        }
        
        full_template = f"""
        {template_text}
        
        Additional instructions: {audience_instructions[audience]}
        
        Structure your response clearly and provide practical examples where appropriate.
        """
        
        return PromptTemplate(
            input_variables=["topic", "topic1", "topic2"],
            template=full_template
        )
    
    # Test dynamic template generation
    test_cases = [
        ("explanation", "simple", "beginner", "artificial intelligence"),
        ("analysis", "detailed", "intermediate", "machine learning algorithms"),
        ("comparison", "technical", "expert", "deep learning", "traditional ML")
    ]
    
    for task_type, complexity, audience, *topics in test_cases:
        print(f"\n--- {task_type.upper()} ({complexity}, {audience}) ---")
        
        template = generate_template_for_task(task_type, complexity, audience)
        
        if task_type == "comparison":
            formatted_prompt = template.format(topic1=topics[0], topic2=topics[1])
        else:
            formatted_prompt = template.format(topic=topics[0])
        
        print(f"Generated Template:\n{formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content[:300]}...\n")

def template_with_conditional_logic():
    """Demonstrate templates with conditional logic."""
    print("=== Conditional Logic Templates ===")
    
    def create_conditional_template(user_input, user_expertise, task_type):
        """Create a template with conditional logic based on user characteristics."""
        
        # Base template
        base_template = "You are a helpful assistant. {task_instruction}"
        
        # Conditional task instructions
        if task_type == "explanation":
            if user_expertise == "beginner":
                task_instruction = """
                Explain {topic} in the simplest possible terms.
                Use analogies and everyday examples.
                Avoid technical jargon.
                """
            elif user_expertise == "intermediate":
                task_instruction = """
                Explain {topic} with some technical detail but keep it accessible.
                Include practical examples and use cases.
                """
            else:  # expert
                task_instruction = """
                Provide a comprehensive technical explanation of {topic}.
                Include advanced concepts, implementation details, and best practices.
                """
        elif task_type == "troubleshooting":
            task_instruction = """
            Help troubleshoot {issue}.
            Provide step-by-step solutions and explain why each step is necessary.
            """
        else:  # general
            task_instruction = "Help with {topic}."
        
        # Add context-specific instructions
        if "code" in user_input.lower():
            task_instruction += "\n\nInclude code examples where appropriate."
        
        if "error" in user_input.lower():
            task_instruction += "\n\nFocus on identifying the root cause and providing solutions."
        
        full_template = base_template.format(task_instruction=task_instruction)
        
        return PromptTemplate(
            input_variables=["topic", "issue"],
            template=full_template
        )
    
    # Test conditional templates
    test_cases = [
        ("Explain machine learning", "beginner", "explanation"),
        ("Debug this Python code", "intermediate", "troubleshooting"),
        ("Advanced neural network architectures", "expert", "explanation")
    ]
    
    for user_input, expertise, task_type in test_cases:
        print(f"\n--- {expertise.upper()} {task_type.upper()} ---")
        
        template = create_conditional_template(user_input, expertise, task_type)
        
        if task_type == "troubleshooting":
            formatted_prompt = template.format(issue=user_input)
        else:
            formatted_prompt = template.format(topic=user_input)
        
        print(f"User Input: {user_input}")
        print(f"Generated Template:\n{formatted_prompt}")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response.content[:300]}...\n")

def template_with_external_data():
    """Demonstrate templates that incorporate external data."""
    print("=== External Data Templates ===")
    
    # Simulate external data sources
    external_data = {
        "user_preferences": {
            "learning_style": "visual",
            "technical_level": "intermediate",
            "preferred_language": "Python"
        },
        "context": {
            "current_project": "web application",
            "team_size": "5 developers",
            "deadline": "2 weeks"
        },
        "constraints": {
            "max_response_length": "500 words",
            "include_code_examples": True,
            "focus_on_practical_applications": True
        }
    }
    
    def create_contextual_template(data):
        """Create a template that incorporates external data."""
        
        # Build dynamic instructions based on data
        style_instruction = ""
        if data["user_preferences"]["learning_style"] == "visual":
            style_instruction = "Include visual analogies and diagrams in your explanation."
        
        level_instruction = f"Target explanation for {data['user_preferences']['technical_level']} level."
        
        project_instruction = f"Relate your explanation to {data['context']['current_project']} development."
        
        constraint_instruction = f"Keep your response under {data['constraints']['max_response_length']}."
        
        if data["constraints"]["include_code_examples"]:
            constraint_instruction += f" Include {data['user_preferences']['preferred_language']} code examples."
        
        template = f"""
        You are a helpful coding assistant.
        
        {style_instruction}
        {level_instruction}
        {project_instruction}
        {constraint_instruction}
        
        User Question: {{question}}
        
        Provide a practical, actionable response that fits the user's context and preferences.
        """
        
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )
    
    # Test contextual template
    template = create_contextual_template(external_data)
    
    question = "How do I implement user authentication in my web app?"
    formatted_prompt = template.format(question=question)
    
    print(f"External Data: {json.dumps(external_data, indent=2)}")
    print(f"Question: {question}")
    print(f"Contextual Template:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content[:400]}...\n")

def main():
    """Run all prompt template examples."""
    print("LangChain Prompt Templates Examples\n")
    print("=" * 50)
    
    try:
        reusable_templates_example()
        pipeline_prompt_example()
        dynamic_template_generation()
        template_with_conditional_logic()
        template_with_external_data()
        
        print("All template examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 