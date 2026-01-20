# 10_full_integration_example.py
# Complete end-to-end example integrating prompts with LangChain LLMs
# This shows how prompts work with actual language models (simulated here)

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# NOTE: In real usage, you would use actual LLM classes like:
# from langchain.chat_models import ChatOpenAI, ChatAnthropic
# from langchain.llms import OpenAI
# 
# For this example, we'll simulate the LLM responses

# ==================== EXAMPLE 1: BASIC CHAIN WITH PROMPT ====================
print("="*70)
print("EXAMPLE 1: BASIC PROMPT + LLM CHAIN")
print("="*70 + "\n")

class SimulatedLLM:
    """Simulated LLM for demonstration (in real use, you'd use actual LLMs)"""
    
    def __call__(self, prompt: str) -> str:
        """Simulate LLM response"""
        print("PROMPT SENT TO LLM:")
        print("-" * 70)
        print(prompt)
        print("-" * 70)
        return "[Simulated LLM Response would appear here]"

# Create prompt template
translation_prompt = PromptTemplate(
    template="Translate the following {source_lang} text to {target_lang}:\n\n{text}",
    input_variables=["source_lang", "target_lang", "text"]
)

# In real usage with actual LLM:
# from langchain.chains import LLMChain
# llm = ChatOpenAI(temperature=0.7)
# chain = LLMChain(llm=llm, prompt=translation_prompt)
# result = chain.run(source_lang="English", target_lang="Spanish", text="Hello, world!")

# Simulated version:
llm = SimulatedLLM()
formatted_prompt = translation_prompt.format(
    source_lang="English",
    target_lang="Spanish",
    text="Hello, how are you today?"
)
response = llm(formatted_prompt)
print("\nRESPONSE:")
print(response)

# ==================== EXAMPLE 2: CHAT MODEL WITH MEMORY ====================
print("\n" + "="*70)
print("EXAMPLE 2: CHAT PROMPT WITH CONVERSATION MEMORY")
print("="*70 + "\n")

# Chat prompt with memory placeholder
chat_prompt_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. You remember previous conversations."),
    MessagesPlaceholder(variable_name="chat_history"),  # This holds conversation history
    ("human", "{input}")
])

# Simulating a conversation with history
conversation_history = [
    {"role": "human", "content": "My name is Alice"},
    {"role": "ai", "content": "Nice to meet you, Alice! How can I help you today?"},
    {"role": "human", "content": "I'm interested in learning Python"},
    {"role": "ai", "content": "Great choice! Python is excellent for beginners. What aspect interests you?"}
]

# In real usage:
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
#
# memory = ConversationBufferMemory(return_messages=True)
# conversation = ConversationChain(
#     llm=ChatOpenAI(),
#     prompt=chat_prompt_with_memory,
#     memory=memory
# )
# response = conversation.predict(input="What was my name again?")

print("Current conversation context being sent:")
print("-" * 70)
print("SYSTEM: You are a helpful AI assistant. You remember previous conversations.")
for msg in conversation_history:
    print(f"{msg['role'].upper()}: {msg['content']}")
print("HUMAN: What was my name again?")
print("-" * 70)
print("\nThe LLM can see the full history and would respond: 'Your name is Alice!'")

# ==================== EXAMPLE 3: STRUCTURED OUTPUT WITH PARSING ====================
print("\n" + "="*70)
print("EXAMPLE 3: STRUCTURED OUTPUT EXTRACTION")
print("="*70 + "\n")

# Define the structure we want
class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    instructions: List[str] = Field(description="Step-by-step cooking instructions")
    prep_time: int = Field(description="Preparation time in minutes")
    cook_time: int = Field(description="Cooking time in minutes")
    servings: int = Field(description="Number of servings")

# Create parser
recipe_parser = PydanticOutputParser(pydantic_object=Recipe)

# Create prompt with parsing instructions
recipe_prompt = PromptTemplate(
    template="""Create a recipe for {dish}.

{format_instructions}

Make it suitable for {dietary_preference} diet with {difficulty} difficulty level.
""",
    input_variables=["dish", "dietary_preference", "difficulty"],
    partial_variables={"format_instructions": recipe_parser.get_format_instructions()}
)

formatted_prompt = recipe_prompt.format(
    dish="vegetable stir fry",
    dietary_preference="vegan",
    difficulty="easy"
)

print("PROMPT WITH STRUCTURED OUTPUT INSTRUCTIONS:")
print("-" * 70)
print(formatted_prompt[:500] + "...")
print("-" * 70)

# Simulated structured response
simulated_json_response = """```json
{
    "name": "Easy Vegan Vegetable Stir Fry",
    "ingredients": [
        "2 cups mixed vegetables",
        "2 tbsp soy sauce",
        "1 tbsp sesame oil",
        "2 cloves garlic",
        "1 tsp ginger"
    ],
    "instructions": [
        "Heat sesame oil in a wok",
        "Add garlic and ginger, stir for 30 seconds",
        "Add vegetables and stir-fry for 5 minutes",
        "Add soy sauce and cook for 2 more minutes"
    ],
    "prep_time": 10,
    "cook_time": 10,
    "servings": 4
}
```"""

# Parse the response
parsed_recipe = recipe_parser.parse(simulated_json_response)
print("\nPARSED RECIPE OBJECT:")
print(f"Name: {parsed_recipe.name}")
print(f"Total Time: {parsed_recipe.prep_time + parsed_recipe.cook_time} minutes")
print(f"Ingredients: {len(parsed_recipe.ingredients)}")
print(f"Type: {type(parsed_recipe)}")

# ==================== EXAMPLE 4: SEQUENTIAL CHAIN ====================
print("\n" + "="*70)
print("EXAMPLE 4: SEQUENTIAL CHAIN (Multi-Step Processing)")
print("="*70 + "\n")

# Step 1: Generate product idea
idea_prompt = PromptTemplate(
    template="Generate a creative product idea for the {industry} industry that solves {problem}.",
    input_variables=["industry", "problem"]
)

# Step 2: Create marketing copy
marketing_prompt = PromptTemplate(
    template="""Based on this product idea: {product_idea}

Create compelling marketing copy including:
- Catchy headline
- 3 key benefits
- Call to action

Target audience: {target_audience}
Tone: {tone}""",
    input_variables=["product_idea", "target_audience", "tone"]
)

# Step 3: Price recommendation
pricing_prompt = PromptTemplate(
    template="""Product: {product_idea}
Marketing Copy: {marketing_copy}

Analyze the market and suggest:
- Recommended price point
- Pricing strategy (premium/competitive/penetration)
- Justification""",
    input_variables=["product_idea", "marketing_copy"]
)

# In real usage:
# from langchain.chains import SequentialChain, LLMChain
# 
# chain1 = LLMChain(llm=llm, prompt=idea_prompt, output_key="product_idea")
# chain2 = LLMChain(llm=llm, prompt=marketing_prompt, output_key="marketing_copy")
# chain3 = LLMChain(llm=llm, prompt=pricing_prompt, output_key="pricing")
#
# overall_chain = SequentialChain(
#     chains=[chain1, chain2, chain3],
#     input_variables=["industry", "problem", "target_audience", "tone"],
#     output_variables=["product_idea", "marketing_copy", "pricing"]
# )
# result = overall_chain({"industry": "fitness", "problem": "home workout motivation"})

print("SEQUENTIAL CHAIN DEMONSTRATION:")
print("-" * 70)

# Step 1
step1_input = {"industry": "fitness", "problem": "lack of motivation for home workouts"}
step1_prompt = idea_prompt.format(**step1_input)
print("STEP 1 - IDEA GENERATION:")
print(step1_prompt)
step1_output = "SmartFit Mirror: An AI-powered smart mirror that provides personalized workout coaching"

print(f"\nOUTPUT: {step1_output}")
print("\n" + "-" * 70)

# Step 2
step2_input = {
    "product_idea": step1_output,
    "target_audience": "busy professionals aged 25-45",
    "tone": "energetic and motivating"
}
step2_prompt = marketing_prompt.format(**step2_input)
print("\nSTEP 2 - MARKETING COPY:")
print(step2_prompt)
step2_output = "Transform Your Home Into Your Personal Gym!"

print(f"\nOUTPUT: {step2_output}")
print("\n" + "-" * 70)

# Step 3
step3_input = {
    "product_idea": step1_output,
    "marketing_copy": step2_output
}
step3_prompt = pricing_prompt.format(**step3_input)
print("\nSTEP 3 - PRICING ANALYSIS:")
print(step3_prompt)

print("\n\nFINAL RESULT: All three outputs combined for complete product launch plan")

# ==================== EXAMPLE 5: ROUTER CHAIN ====================
print("\n" + "="*70)
print("EXAMPLE 5: ROUTER CHAIN (Conditional Routing)")
print("="*70 + "\n")

# Different prompts for different types of questions
physics_prompt = PromptTemplate(
    template="""You are a physics expert. Explain this concept clearly with formulas and examples:

{question}

Include:
- Core physics principles
- Relevant equations
- Practical examples""",
    input_variables=["question"]
)

biology_prompt = PromptTemplate(
    template="""You are a biology expert. Explain this concept with emphasis on biological processes:

{question}

Include:
- Biological mechanisms
- Cellular or molecular details
- Real-world relevance""",
    input_variables=["question"]
)

math_prompt = PromptTemplate(
    template="""You are a mathematics expert. Provide a rigorous mathematical explanation:

{question}

Include:
- Mathematical definitions
- Proofs or derivations
- Worked examples""",
    input_variables=["question"]
)

general_prompt = PromptTemplate(
    template="""Provide a clear, comprehensive explanation of:

{question}

Use simple language and relevant examples.""",
    input_variables=["question"]
)

# Router logic (in real use, this would be done by the LLM or a classifier)
def route_question(question: str) -> str:
    """Determine which prompt to use based on question content"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['force', 'energy', 'motion', 'velocity', 'quantum']):
        return 'physics'
    elif any(word in question_lower for word in ['cell', 'dna', 'organism', 'evolution', 'protein']):
        return 'biology'
    elif any(word in question_lower for word in ['equation', 'derivative', 'integral', 'theorem', 'proof']):
        return 'math'
    else:
        return 'general'

# Example questions
questions = [
    "What is Newton's second law of motion?",
    "How does DNA replication work?",
    "What is the Pythagorean theorem?"
]

prompt_map = {
    'physics': physics_prompt,
    'biology': biology_prompt,
    'math': math_prompt,
    'general': general_prompt
}

print("ROUTER CHAIN DEMONSTRATION:")
print("-" * 70)

for question in questions:
    route = route_question(question)
    selected_prompt = prompt_map[route]
    formatted = selected_prompt.format(question=question)
    
    print(f"\nQUESTION: {question}")
    print(f"ROUTED TO: {route.upper()}")
    print(f"PROMPT USED:")
    print(formatted)
    print("-" * 70)

# ==================== BEST PRACTICES SUMMARY ====================
print("\n" + "="*70)
print("INTEGRATION BEST PRACTICES")
print("="*70)
print("""
1. PROMPT DESIGN:
   - Be specific and clear in instructions
   - Use examples (few-shot learning) for complex tasks
   - Include format instructions for structured outputs

2. CHAIN COMPOSITION:
   - Break complex tasks into sequential steps
   - Use routers for conditional logic
   - Implement proper error handling

3. MEMORY MANAGEMENT:
   - Use ConversationBufferMemory for chat applications
   - Consider ConversationSummaryMemory for long conversations
   - Clear memory when starting new conversations

4. OUTPUT PARSING:
   - Use Pydantic models for type-safe structured outputs
   - Always validate parsed outputs
   - Handle parsing errors gracefully

5. OPTIMIZATION:
   - Cache frequently used prompts
   - Minimize token usage with concise prompts
   - Use appropriate temperature settings

6. TESTING:
   - Test prompts with various inputs
   - Validate edge cases
   - Monitor LLM costs and performance

REAL USAGE EXAMPLE:
```python
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain

# Initialize LLM
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Create chain
chain = LLMChain(llm=llm, prompt=your_prompt)

# Run chain
result = chain.run(your_input_variables)
```
""")