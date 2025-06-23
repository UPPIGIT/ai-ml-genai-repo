"""
LangChain Chain Prompts Examples
This file demonstrates multi-step reasoning and complex workflow prompts.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts.pipeline import PipelinePromptTemplate

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def simple_sequential_chain():
    """Demonstrate simple sequential chain for multi-step processing."""
    print("=== Simple Sequential Chain ===")
    
    # Step 1: Generate a story idea
    story_idea_prompt = PromptTemplate(
        input_variables=["genre", "theme"],
        template="""
        Generate a creative story idea for a {genre} story with the theme of {theme}.
        Provide a brief plot summary in 2-3 sentences.
        """
    )
    
    # Step 2: Expand the story idea into a detailed outline
    outline_prompt = PromptTemplate(
        input_variables=["story_idea"],
        template="""
        Based on this story idea: {story_idea}
        
        Create a detailed story outline with:
        1. Introduction/Setup
        2. Rising Action
        3. Climax
        4. Falling Action
        5. Resolution
        
        Make it engaging and well-structured.
        """
    )
    
    # Step 3: Write the opening scene
    opening_prompt = PromptTemplate(
        input_variables=["outline"],
        template="""
        Based on this story outline: {outline}
        
        Write an engaging opening scene (first 2-3 paragraphs) that hooks the reader.
        Focus on setting the mood and introducing the main character or situation.
        """
    )
    
    # Create the chains
    story_idea_chain = LLMChain(llm=llm, prompt=story_idea_prompt)
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
    opening_chain = LLMChain(llm=llm, prompt=opening_prompt)
    
    # Create the sequential chain
    full_chain = SimpleSequentialChain(
        chains=[story_idea_chain, outline_chain, opening_chain],
        verbose=True
    )
    
    # Test the chain
    result = full_chain.run({
        "genre": "science fiction",
        "theme": "artificial intelligence"
    })
    
    print(f"Final Result:\n{result}\n")

def complex_sequential_chain():
    """Demonstrate complex sequential chain with multiple inputs and outputs."""
    print("=== Complex Sequential Chain ===")
    
    # Step 1: Analyze a problem
    problem_analysis_prompt = PromptTemplate(
        input_variables=["problem_description"],
        template="""
        Analyze the following problem:
        {problem_description}
        
        Provide:
        1. Root cause analysis
        2. Impact assessment
        3. Stakeholders affected
        4. Priority level (High/Medium/Low)
        """
    )
    
    # Step 2: Generate solutions
    solution_generation_prompt = PromptTemplate(
        input_variables=["problem_analysis", "constraints"],
        template="""
        Based on this problem analysis:
        {problem_analysis}
        
        And considering these constraints:
        {constraints}
        
        Generate 3 potential solutions with:
        1. Implementation steps
        2. Required resources
        3. Timeline estimate
        4. Risk assessment
        """
    )
    
    # Step 3: Evaluate solutions
    solution_evaluation_prompt = PromptTemplate(
        input_variables=["solutions", "criteria"],
        template="""
        Evaluate these solutions:
        {solutions}
        
        Against these criteria:
        {criteria}
        
        Provide:
        1. Pros and cons for each solution
        2. Cost-benefit analysis
        3. Recommended solution with justification
        4. Implementation roadmap
        """
    )
    
    # Create the chains
    analysis_chain = LLMChain(
        llm=llm, 
        prompt=problem_analysis_prompt, 
        output_key="problem_analysis"
    )
    
    solution_chain = LLMChain(
        llm=llm, 
        prompt=solution_generation_prompt, 
        output_key="solutions"
    )
    
    evaluation_chain = LLMChain(
        llm=llm, 
        prompt=solution_evaluation_prompt, 
        output_key="evaluation"
    )
    
    # Create the sequential chain
    full_chain = SequentialChain(
        chains=[analysis_chain, solution_chain, evaluation_chain],
        input_variables=["problem_description", "constraints", "criteria"],
        output_variables=["problem_analysis", "solutions", "evaluation"],
        verbose=True
    )
    
    # Test the chain
    inputs = {
        "problem_description": "Our e-commerce website is experiencing slow loading times during peak hours, leading to customer complaints and lost sales.",
        "constraints": "Budget: $50,000, Timeline: 3 months, Team: 5 developers",
        "criteria": "Performance improvement, Cost-effectiveness, User experience, Scalability"
    }
    
    result = full_chain(inputs)
    
    print("Problem Analysis:")
    print(result["problem_analysis"])
    print("\nSolutions:")
    print(result["solutions"])
    print("\nEvaluation:")
    print(result["evaluation"])
    print()

def pipeline_prompt_chain():
    """Demonstrate pipeline prompt templates for complex workflows."""
    print("=== Pipeline Prompt Chain ===")
    
    # Base template for research
    research_template = """
    You are a research analyst. Research the following topic:
    {topic}
    
    Provide:
    1. Key facts and statistics
    2. Current trends
    3. Major players or stakeholders
    4. Recent developments
    """
    
    # Analysis template
    analysis_template = """
    Based on this research:
    {research}
    
    Analyze the topic from a {perspective} perspective:
    1. Strengths and opportunities
    2. Weaknesses and threats
    3. Market positioning
    4. Future outlook
    """
    
    # Recommendation template
    recommendation_template = """
    Based on this analysis:
    {analysis}
    
    Provide strategic recommendations for {stakeholder}:
    1. Immediate actions (next 30 days)
    2. Short-term strategies (3-6 months)
    3. Long-term vision (1-2 years)
    4. Risk mitigation strategies
    """
    
    # Create the pipeline
    research_prompt = PromptTemplate(
        input_variables=["topic"],
        template=research_template
    )
    
    analysis_prompt = PromptTemplate(
        input_variables=["research", "perspective"],
        template=analysis_template
    )
    
    recommendation_prompt = PromptTemplate(
        input_variables=["analysis", "stakeholder"],
        template=recommendation_template
    )
    
    # Create pipeline prompts
    analysis_pipeline = PipelinePromptTemplate(
        final_prompt=analysis_prompt,
        pipeline_prompts=[("research", research_prompt)]
    )
    
    recommendation_pipeline = PipelinePromptTemplate(
        final_prompt=recommendation_prompt,
        pipeline_prompts=[("analysis", analysis_pipeline)]
    )
    
    # Test the pipeline
    topic = "artificial intelligence in healthcare"
    perspective = "business"
    stakeholder = "healthcare providers"
    
    formatted_prompt = recommendation_pipeline.format(
        topic=topic,
        perspective=perspective,
        stakeholder=stakeholder
    )
    
    print(f"Topic: {topic}")
    print(f"Perspective: {perspective}")
    print(f"Stakeholder: {stakeholder}")
    print(f"Pipeline Prompt:\n{formatted_prompt}")
    
    response = llm.invoke(formatted_prompt)
    print(f"Response: {response.content}\n")

def multi_agent_chain():
    """Demonstrate multi-agent chain with different specialized roles."""
    print("=== Multi-Agent Chain ===")
    
    # Agent 1: Technical Expert
    technical_prompt = PromptTemplate(
        input_variables=["requirement"],
        template="""
        You are a technical expert. Analyze this requirement:
        {requirement}
        
        Provide:
        1. Technical feasibility assessment
        2. Required technologies and tools
        3. Architecture recommendations
        4. Technical challenges and solutions
        5. Implementation complexity (Low/Medium/High)
        """
    )
    
    # Agent 2: Business Analyst
    business_prompt = PromptTemplate(
        input_variables=["technical_analysis", "business_context"],
        template="""
        Based on this technical analysis:
        {technical_analysis}
        
        And business context:
        {business_context}
        
        Provide business analysis:
        1. ROI assessment
        2. Market opportunity
        3. Competitive advantage
        4. Business risks
        5. Success metrics
        """
    )
    
    # Agent 3: Project Manager
    project_prompt = PromptTemplate(
        input_variables=["technical_analysis", "business_analysis"],
        template="""
        Based on:
        Technical Analysis: {technical_analysis}
        Business Analysis: {business_analysis}
        
        Create a project plan:
        1. Project phases and timeline
        2. Resource requirements
        3. Risk management plan
        4. Success criteria
        5. Go/No-go decision with justification
        """
    )
    
    # Create the chains
    technical_chain = LLMChain(
        llm=llm, 
        prompt=technical_prompt, 
        output_key="technical_analysis"
    )
    
    business_chain = LLMChain(
        llm=llm, 
        prompt=business_prompt, 
        output_key="business_analysis"
    )
    
    project_chain = LLMChain(
        llm=llm, 
        prompt=project_prompt, 
        output_key="project_plan"
    )
    
    # Create the sequential chain
    full_chain = SequentialChain(
        chains=[technical_chain, business_chain, project_chain],
        input_variables=["requirement", "business_context"],
        output_variables=["technical_analysis", "business_analysis", "project_plan"],
        verbose=True
    )
    
    # Test the chain
    inputs = {
        "requirement": "Build a machine learning system to predict customer churn for our SaaS platform",
        "business_context": "We have 10,000 customers, $2M ARR, and want to reduce churn by 20%"
    }
    
    result = full_chain(inputs)
    
    print("Technical Analysis:")
    print(result["technical_analysis"])
    print("\nBusiness Analysis:")
    print(result["business_analysis"])
    print("\nProject Plan:")
    print(result["project_plan"])
    print()

def iterative_refinement_chain():
    """Demonstrate iterative refinement through multiple chain steps."""
    print("=== Iterative Refinement Chain ===")
    
    # Step 1: Initial draft
    draft_prompt = PromptTemplate(
        input_variables=["topic", "audience"],
        template="""
        Write an initial draft about {topic} for {audience}.
        Focus on getting the main ideas down.
        """
    )
    
    # Step 2: Content enhancement
    enhancement_prompt = PromptTemplate(
        input_variables=["draft", "style_guide"],
        template="""
        Enhance this draft:
        {draft}
        
        Following this style guide:
        {style_guide}
        
        Add more detail, examples, and improve clarity.
        """
    )
    
    # Step 3: Structure improvement
    structure_prompt = PromptTemplate(
        input_variables=["enhanced_content"],
        template="""
        Improve the structure and flow of this content:
        {enhanced_content}
        
        Ensure logical progression, clear sections, and smooth transitions.
        """
    )
    
    # Step 4: Final polish
    polish_prompt = PromptTemplate(
        input_variables=["structured_content", "tone"],
        template="""
        Give this content a final polish:
        {structured_content}
        
        Adjust the tone to be {tone}.
        Fix any grammar, spelling, or style issues.
        Make it engaging and professional.
        """
    )
    
    # Create the chains
    draft_chain = LLMChain(llm=llm, prompt=draft_prompt, output_key="draft")
    enhancement_chain = LLMChain(llm=llm, prompt=enhancement_prompt, output_key="enhanced_content")
    structure_chain = LLMChain(llm=llm, prompt=structure_prompt, output_key="structured_content")
    polish_chain = LLMChain(llm=llm, prompt=polish_prompt, output_key="final_content")
    
    # Create the sequential chain
    full_chain = SequentialChain(
        chains=[draft_chain, enhancement_chain, structure_chain, polish_chain],
        input_variables=["topic", "audience", "style_guide", "tone"],
        output_variables=["draft", "enhanced_content", "structured_content", "final_content"],
        verbose=True
    )
    
    # Test the chain
    inputs = {
        "topic": "the benefits of remote work",
        "audience": "business leaders",
        "style_guide": "Professional, data-driven, actionable insights",
        "tone": "confident and persuasive"
    }
    
    result = full_chain(inputs)
    
    print("Draft:")
    print(result["draft"])
    print("\nEnhanced Content:")
    print(result["enhanced_content"])
    print("\nStructured Content:")
    print(result["structured_content"])
    print("\nFinal Content:")
    print(result["final_content"])
    print()

def main():
    """Run all chain prompt examples."""
    print("LangChain Chain Prompts Examples\n")
    print("=" * 50)
    
    try:
        simple_sequential_chain()
        complex_sequential_chain()
        pipeline_prompt_chain()
        multi_agent_chain()
        iterative_refinement_chain()
        
        print("All chain examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 