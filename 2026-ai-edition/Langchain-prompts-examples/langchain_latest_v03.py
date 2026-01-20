# 13_langchain_v0.3_latest_examples.py
# Examples using LangChain 0.3+ with LCEL (LangChain Expression Language)
# This is the MODERN way to use LangChain with the latest patterns

"""
INSTALLATION (Latest versions):
pip install langchain==0.3.0 langchain-openai==0.2.0 langchain-anthropic==0.3.0 
pip install langchain-core==0.3.0 langchain-community==0.3.0 python-dotenv

BREAKING CHANGES IN 0.3+:
- LLMChain is deprecated → Use LCEL (pipe operator |)
- New import paths for chat models
- Runnable interface for all chains
- Better streaming support
- Improved async support
"""

import os
from dotenv import load_dotenv

# Modern imports for LangChain 0.3+
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Dict

load_dotenv()

# ==================== EXAMPLE 1: LCEL BASICS (PIPE OPERATOR) ====================
print("="*70)
print("EXAMPLE 1: LCEL - The Modern LangChain Way")
print("="*70 + "\n")

def example1_lcel_basics():
    """
    LCEL (LangChain Expression Language) uses the pipe operator (|)
    This is the NEW recommended way instead of LLMChain
    """
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}")
    ])
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Chain components using LCEL (pipe operator)
    # This is the MODERN way - no more LLMChain!
    chain = prompt | llm | output_parser
    
    # Invoke the chain
    result = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?"
    })
    
    print("LCEL CHAIN RESULT:")
    print(result)
    print("\n" + "="*70 + "\n")
    
    # You can also use batch for multiple inputs
    results = chain.batch([
        {"input_language": "English", "output_language": "Spanish", "text": "Good morning"},
        {"input_language": "English", "output_language": "German", "text": "Thank you"}
    ])
    
    print("BATCH RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example1_lcel_basics()

# ==================== EXAMPLE 2: STREAMING WITH LCEL ====================
print("EXAMPLE 2: STREAMING WITH LCEL")
print("="*70 + "\n")

def example2_streaming():
    """Modern streaming approach with LCEL"""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative storyteller."),
        ("human", "Write a short story about {topic}")
    ])
    
    # Create streaming chain
    chain = prompt | llm | StrOutputParser()
    
    print("STREAMING OUTPUT:")
    print("-"*70)
    
    # Stream method for real-time output
    for chunk in chain.stream({"topic": "a robot learning to paint"}):
        print(chunk, end="", flush=True)
    
    print("\n" + "-"*70 + "\n")
    print("="*70 + "\n")

# Uncomment to run:
# example2_streaming()

# ==================== EXAMPLE 3: STRUCTURED OUTPUT WITH LCEL ====================
print("EXAMPLE 3: STRUCTURED OUTPUT (LATEST PATTERN)")
print("="*70 + "\n")

def example3_structured_output():
    """Modern structured output using Pydantic with LCEL"""
    
    # Define output structure
    class Recipe(BaseModel):
        """Recipe structure"""
        name: str = Field(description="Recipe name")
        ingredients: List[str] = Field(description="List of ingredients")
        steps: List[str] = Field(description="Cooking steps")
        prep_time: int = Field(description="Prep time in minutes")
        difficulty: str = Field(description="easy, medium, or hard")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    # Create prompt with structured output instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a chef. Generate recipes in the requested format."),
        ("human", "Create a recipe for {dish}. {format_instructions}")
    ])
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=Recipe)
    
    # Modern LCEL chain with structured output
    chain = (
        {
            "dish": RunnablePassthrough(),
            "format_instructions": lambda x: parser.get_format_instructions()
        }
        | prompt 
        | llm 
        | parser
    )
    
    # Invoke and get structured output
    result = chain.invoke("chocolate chip cookies")
    
    print("STRUCTURED OUTPUT:")
    print(f"Recipe: {result.name}")
    print(f"Difficulty: {result.difficulty}")
    print(f"Prep Time: {result.prep_time} minutes")
    print(f"Ingredients ({len(result.ingredients)}):")
    for ingredient in result.ingredients:
        print(f"  - {ingredient}")
    print(f"Steps ({len(result.steps)}):")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example3_structured_output()

# ==================== EXAMPLE 4: PARALLEL EXECUTION ====================
print("EXAMPLE 4: PARALLEL EXECUTION WITH RUNNABLEPARALLEL")
print("="*70 + "\n")

def example4_parallel_execution():
    """Execute multiple chains in parallel - LCEL feature"""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create different chains for different tasks
    joke_chain = (
        ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        | llm
        | StrOutputParser()
    )
    
    poem_chain = (
        ChatPromptTemplate.from_template("Write a haiku about {topic}")
        | llm
        | StrOutputParser()
    )
    
    fact_chain = (
        ChatPromptTemplate.from_template("Give me an interesting fact about {topic}")
        | llm
        | StrOutputParser()
    )
    
    # Run all chains in parallel
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        poem=poem_chain,
        fact=fact_chain
    )
    
    result = parallel_chain.invoke({"topic": "programming"})
    
    print("PARALLEL EXECUTION RESULTS:")
    print("-"*70)
    print(f"JOKE:\n{result['joke']}\n")
    print(f"POEM:\n{result['poem']}\n")
    print(f"FACT:\n{result['fact']}\n")
    print("-"*70 + "\n")
    print("="*70 + "\n")

# Uncomment to run:
# example4_parallel_execution()

# ==================== EXAMPLE 5: CONDITIONAL ROUTING ====================
print("EXAMPLE 5: CONDITIONAL ROUTING (BRANCHING)")
print("="*70 + "\n")

def example5_conditional_routing():
    """Route to different chains based on input - modern pattern"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    # Different prompts for different categories
    technical_prompt = ChatPromptTemplate.from_template(
        "Provide a technical explanation of {question} with code examples if relevant."
    )
    
    simple_prompt = ChatPromptTemplate.from_template(
        "Explain {question} in simple terms that a beginner would understand."
    )
    
    # Create chains
    technical_chain = technical_prompt | llm | StrOutputParser()
    simple_chain = simple_prompt | llm | StrOutputParser()
    
    # Router function
    def route_question(input_dict):
        """Determine which chain to use"""
        question = input_dict["question"].lower()
        
        # Check if question contains technical keywords
        technical_keywords = ["algorithm", "complexity", "implementation", "code", "optimize"]
        if any(keyword in question for keyword in technical_keywords):
            return technical_chain
        else:
            return simple_chain
    
    # Create routing chain
    routing_chain = RunnableLambda(route_question)
    
    # Full chain with routing
    chain = (
        RunnablePassthrough()
        | routing_chain
    )
    
    # Test with different questions
    questions = [
        {"question": "What is a sorting algorithm and how do I implement one?"},
        {"question": "What is machine learning?"}
    ]
    
    print("CONDITIONAL ROUTING RESULTS:")
    print("-"*70)
    for q in questions:
        print(f"\nQUESTION: {q['question']}")
        result = chain.invoke(q)
        print(f"ANSWER: {result[:200]}...")
        print("-"*70)
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example5_conditional_routing()

# ==================== EXAMPLE 6: MEMORY WITH LCEL ====================
print("EXAMPLE 6: CONVERSATION MEMORY (MODERN APPROACH)")
print("="*70 + "\n")

def example6_memory():
    """Implement conversation memory with LCEL"""
    
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create prompt with message history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember previous messages in the conversation."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Store for conversation history
    store = {}
    
    def get_session_history(session_id: str):
        """Get or create message history for a session"""
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Wrap chain with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Have a conversation
    session_id = "user_123"
    
    print("CONVERSATION WITH MEMORY:")
    print("-"*70)
    
    messages = [
        "Hi! My name is Alice and I love Python programming.",
        "What programming language did I just mention?",
        "What's my name?"
    ]
    
    for msg in messages:
        print(f"\nHUMAN: {msg}")
        response = chain_with_history.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response}")
    
    print("\n" + "-"*70 + "\n")
    print("="*70 + "\n")

# Uncomment to run:
# example6_memory()

# ==================== EXAMPLE 7: ASYNC OPERATIONS ====================
print("EXAMPLE 7: ASYNC OPERATIONS WITH LCEL")
print("="*70 + "\n")

async def example7_async():
    """Modern async support in LangChain 0.3+"""
    
    import asyncio
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("Write a one-line description of {topic}")
    
    chain = prompt | llm | StrOutputParser()
    
    # Async invoke
    result = await chain.ainvoke({"topic": "quantum computing"})
    print(f"ASYNC RESULT: {result}\n")
    
    # Async batch processing
    topics = [
        {"topic": "artificial intelligence"},
        {"topic": "blockchain"},
        {"topic": "cloud computing"}
    ]
    
    results = await chain.abatch(topics)
    
    print("ASYNC BATCH RESULTS:")
    for topic, result in zip(topics, results):
        print(f"{topic['topic']}: {result}")
    
    # Async streaming
    print("\nASYNC STREAMING:")
    print("-"*70)
    async for chunk in chain.astream({"topic": "space exploration"}):
        print(chunk, end="", flush=True)
    print("\n" + "-"*70 + "\n")

def run_async_example():
    """Wrapper to run async example"""
    import asyncio
    asyncio.run(example7_async())

# Uncomment to run:
# run_async_example()

# ==================== EXAMPLE 8: CHAINING WITH RUNNABLE LAMBDA ====================
print("EXAMPLE 8: CUSTOM PROCESSING WITH RUNNABLELAMBDA")
print("="*70 + "\n")

def example8_custom_processing():
    """Add custom processing steps using RunnableLambda"""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template("List 5 {items}")
    
    # Custom processing function
    def format_list(text: str) -> Dict[str, any]:
        """Extract and format list items"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        items = [line.lstrip('123456789.- ') for line in lines if line]
        return {
            "count": len(items),
            "items": items,
            "formatted": "\n".join(f"✓ {item}" for item in items)
        }
    
    # Chain with custom processing
    chain = (
        prompt 
        | llm 
        | StrOutputParser() 
        | RunnableLambda(format_list)  # Custom processing step
    )
    
    result = chain.invoke({"items": "programming languages"})
    
    print("CUSTOM PROCESSING RESULT:")
    print(f"Count: {result['count']}")
    print(f"\nFormatted List:\n{result['formatted']}")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example8_custom_processing()

# ==================== EXAMPLE 9: REAL-WORLD APPLICATION ====================
print("EXAMPLE 9: COMPLETE APPLICATION - BLOG POST GENERATOR")
print("="*70 + "\n")

def example9_blog_generator():
    """Complete blog post generation system using LCEL"""
    
    class BlogPost(BaseModel):
        title: str = Field(description="Catchy blog post title")
        introduction: str = Field(description="Engaging introduction")
        sections: List[Dict[str, str]] = Field(description="List of sections with headers and content")
        conclusion: str = Field(description="Strong conclusion")
        keywords: List[str] = Field(description="SEO keywords")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Step 1: Generate outline
    outline_prompt = ChatPromptTemplate.from_template(
        "Create a detailed outline for a blog post about {topic}. Target audience: {audience}"
    )
    
    # Step 2: Generate full content
    content_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional content writer."),
        ("human", "Based on this outline:\n{outline}\n\nWrite a complete blog post. {format_instructions}")
    ])
    
    parser = PydanticOutputParser(pydantic_object=BlogPost)
    
    # Create LCEL chain
    chain = (
        {
            "outline": outline_prompt | llm | StrOutputParser(),
            "format_instructions": lambda x: parser.get_format_instructions()
        }
        | content_prompt
        | llm
        | parser
    )
    
    # Generate blog post
    result = chain.invoke({
        "topic": "The Future of Remote Work",
        "audience": "business professionals"
    })
    
    print("GENERATED BLOG POST:")
    print("-"*70)
    print(f"Title: {result.title}\n")
    print(f"Introduction:\n{result.introduction}\n")
    print(f"Sections ({len(result.sections)}):")
    for i, section in enumerate(result.sections, 1):
        print(f"\n{i}. {section.get('header', 'Section')}")
        print(f"   {section.get('content', '')[:100]}...")
    print(f"\nConclusion:\n{result.conclusion[:200]}...")
    print(f"\nSEO Keywords: {', '.join(result.keywords)}")
    print("-"*70 + "\n")

# Uncomment to run:
# example9_blog_generator()

# ==================== EXAMPLE 10: ERROR HANDLING IN LCEL ====================
print("EXAMPLE 10: ERROR HANDLING AND FALLBACKS")
print("="*70 + "\n")

def example10_error_handling():
    """Modern error handling with fallback chains"""
    
    from langchain_core.runnables import RunnableFallback
    
    # Primary chain (might fail)
    primary_llm = ChatOpenAI(model="gpt-4", temperature=0.7, max_retries=1)
    primary_chain = (
        ChatPromptTemplate.from_template("Explain {concept}")
        | primary_llm
        | StrOutputParser()
    )
    
    # Fallback chain (simpler, more reliable)
    fallback_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    fallback_chain = (
        ChatPromptTemplate.from_template("Give a simple explanation of {concept}")
        | fallback_llm
        | StrOutputParser()
    )
    
    # Chain with fallback
    chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])
    
    try:
        result = chain_with_fallback.invoke({"concept": "quantum entanglement"})
        print("RESULT (with fallback handling):")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example10_error_handling()

# ==================== SUMMARY ====================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                 LANGCHAIN 0.3+ MODERN PATTERNS                       ║
╚══════════════════════════════════════════════════════════════════════╝

KEY CHANGES IN LANGCHAIN 0.3+:
✓ LCEL (pipe operator |) replaces LLMChain
✓ Runnable interface for everything
✓ Better streaming and async support
✓ New import paths
✓ More composable and flexible

EXAMPLES COVERED:
1. LCEL Basics - Modern chaining with |
2. Streaming - Real-time output
3. Structured Output - Type-safe responses
4. Parallel Execution - Run multiple chains
5. Conditional Routing - Branch logic
6. Memory - Conversation context
7. Async Operations - Non-blocking calls
8. Custom Processing - RunnableLambda
9. Real Application - Blog generator
10. Error Handling - Fallbacks

MIGRATION FROM OLD TO NEW:
❌ OLD: LLMChain(llm=llm, prompt=prompt)
✅ NEW: prompt | llm | parser

❌ OLD: chain.run(input="hello")
✅ NEW: chain.invoke({"input": "hello"})

❌ OLD: ConversationChain with memory
✅ NEW: RunnableWithMessageHistory

TO RUN EXAMPLES:
1. Uncomment any example function above
2. Run: python 13_langchain_v0.3_latest_examples.py
3. Make sure .env has API keys
""")