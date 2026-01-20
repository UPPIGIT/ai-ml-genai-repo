# 06_pipeline_prompts.py
# Pipeline prompts allow you to compose multiple prompts together
# Useful for complex multi-step tasks where one prompt builds on another

from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

# Example 1: Basic Pipeline - Breaking down a complex task
# Step 1: Create individual prompt components

# Component 1: Introduction prompt
intro_template = PromptTemplate.from_template("""
You are an expert {role}.
""")

# Component 2: Context prompt
context_template = PromptTemplate.from_template("""
Here is some background information:
{context}
""")

# Component 3: Task prompt
task_template = PromptTemplate.from_template("""
Please complete the following task:
{task}
""")

# Component 4: Format instructions
format_template = PromptTemplate.from_template("""
Format your response as follows:
{format_instructions}
""")

# Final template that combines all components
final_template = PromptTemplate.from_template("""
{introduction}
{context}
{task}
{format}

Please begin:
""")

# Create the pipeline
pipeline_prompt1 = PipelinePromptTemplate(
    final_prompt=final_template,
    pipeline_prompts=[
        ("introduction", intro_template),
        ("context", context_template),
        ("task", task_template),
        ("format", format_template),
    ]
)

# Format the entire pipeline
prompt1 = pipeline_prompt1.format(
    role="technical writer",
    context="We are documenting a new API for developers",
    task="Write an introduction section for the API documentation",
    format_instructions="- Use clear headings\n- Include code examples\n- Keep it concise"
)

print("Example 1 - Basic Pipeline:")
print(prompt1)
print("\n" + "="*50 + "\n")

# Example 2: Layered Analysis Pipeline
# Build a prompt that guides through multiple analytical steps

# Layer 1: Data presentation
data_layer = PromptTemplate.from_template("""
DATA TO ANALYZE:
{data}
""")

# Layer 2: Analysis framework
framework_layer = PromptTemplate.from_template("""
ANALYSIS FRAMEWORK:
Apply the following methodology:
1. {step1}
2. {step2}
3. {step3}
""")

# Layer 3: Output requirements
output_layer = PromptTemplate.from_template("""
OUTPUT REQUIREMENTS:
- Confidence level: {confidence_level}
- Length: {length}
- Include: {include_items}
""")

# Combine into analysis pipeline
analysis_final = PromptTemplate.from_template("""
{data_section}

{framework_section}

{output_section}

Begin your analysis:
""")

analysis_pipeline = PipelinePromptTemplate(
    final_prompt=analysis_final,
    pipeline_prompts=[
        ("data_section", data_layer),
        ("framework_section", framework_layer),
        ("output_section", output_layer),
    ]
)

prompt2 = analysis_pipeline.format(
    data="Sales: Q1=$100k, Q2=$150k, Q3=$120k, Q4=$180k",
    step1="Identify trends",
    step2="Calculate growth rates",
    step3="Make predictions",
    confidence_level="High",
    length="2-3 paragraphs",
    include_items="charts and key metrics"
)

print("Example 2 - Layered Analysis Pipeline:")
print(prompt2)
print("\n" + "="*50 + "\n")

# Example 3: Conditional Pipeline
# Different components based on task type

def create_task_pipeline(task_type):
    """Create different pipelines based on task type"""
    
    # Common components
    base_template = PromptTemplate.from_template("Task Type: {task_type}\n")
    
    if task_type == "creative":
        specific_template = PromptTemplate.from_template("""
Creative Guidelines:
- Be imaginative and original
- Use vivid descriptions
- Topic: {topic}
""")
    else:  # analytical
        specific_template = PromptTemplate.from_template("""
Analytical Guidelines:
- Use data and facts
- Be objective and precise
- Topic: {topic}
""")
    
    final = PromptTemplate.from_template("""
{base}
{specific}

Execute the task:
""")
    
    return PipelinePromptTemplate(
        final_prompt=final,
        pipeline_prompts=[
            ("base", base_template),
            ("specific", specific_template),
        ]
    )

# Create and use creative pipeline
creative_pipeline = create_task_pipeline("creative")
prompt3 = creative_pipeline.format(task_type="creative", topic="Write a story about time travel")

print("Example 3 - Conditional Pipeline (Creative):")
print(prompt3)
print("\n" + "="*50 + "\n")

# Example 4: Nested Pipeline with reusable components
# Build complex prompts from smaller, reusable pieces

# Reusable component: Persona
persona_template = PromptTemplate.from_template("""
You are a {persona_type} with {years} years of experience in {field}.
""")

# Reusable component: Constraints
constraints_template = PromptTemplate.from_template("""
Constraints:
- Time available: {time}
- Resources: {resources}
- Priority: {priority}
""")

# Reusable component: Success criteria
success_template = PromptTemplate.from_template("""
Success Criteria:
{criteria}
""")

# Main task template
main_task_template = PromptTemplate.from_template("""
Main Objective:
{objective}
""")

# Combine everything
nested_final = PromptTemplate.from_template("""
{persona}

{main_task}

{constraints}

{success_criteria}

Proceed with your recommendation:
""")

nested_pipeline = PipelinePromptTemplate(
    final_prompt=nested_final,
    pipeline_prompts=[
        ("persona", persona_template),
        ("main_task", main_task_template),
        ("constraints", constraints_template),
        ("success_criteria", success_template),
    ]
)

prompt4 = nested_pipeline.format(
    persona_type="project manager",
    years="10",
    field="software development",
    objective="Launch a new mobile app within 6 months",
    time="6 months",
    resources="Team of 5 developers, $200k budget",
    priority="High",
    criteria="- App in app stores\n- 10,000 downloads in first month\n- 4+ star rating"
)

print("Example 4 - Nested Reusable Pipeline:")
print(prompt4)
print("\n" + "="*50 + "\n")

# Example 5: Multi-stage reasoning pipeline
# Guide the AI through a structured thinking process

# Stage 1: Problem understanding
understanding_template = PromptTemplate.from_template("""
STAGE 1 - UNDERSTAND THE PROBLEM:
Problem: {problem}
Key elements: {key_elements}
""")

# Stage 2: Solution brainstorming
brainstorm_template = PromptTemplate.from_template("""
STAGE 2 - BRAINSTORM SOLUTIONS:
Consider these approaches:
{approaches}
""")

# Stage 3: Evaluation
evaluation_template = PromptTemplate.from_template("""
STAGE 3 - EVALUATE OPTIONS:
Evaluation criteria:
{criteria}
""")

# Stage 4: Recommendation
recommendation_template = PromptTemplate.from_template("""
STAGE 4 - MAKE RECOMMENDATION:
Provide your final recommendation with justification.
""")

# Complete reasoning pipeline
reasoning_final = PromptTemplate.from_template("""
STRUCTURED REASONING PROCESS:

{stage1}

{stage2}

{stage3}

{stage4}
""")

reasoning_pipeline = PipelinePromptTemplate(
    final_prompt=reasoning_final,
    pipeline_prompts=[
        ("stage1", understanding_template),
        ("stage2", brainstorm_template),
        ("stage3", evaluation_template),
        ("stage4", recommendation_template),
    ]
)

prompt5 = reasoning_pipeline.format(
    problem="Website loading slowly",
    key_elements="Images, Database queries, Server location",
    approaches="1. Image optimization\n2. Database indexing\n3. CDN implementation\n4. Code minification",
    criteria="- Cost\n- Implementation time\n- Expected improvement"
)

print("Example 5 - Multi-Stage Reasoning Pipeline:")
print(prompt5)