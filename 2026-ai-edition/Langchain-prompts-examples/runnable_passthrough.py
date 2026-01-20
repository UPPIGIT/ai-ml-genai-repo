# 15_understanding_runnable_passthrough.py
# Deep dive into RunnablePassthrough - one of the most important LCEL components
# RunnablePassthrough passes data through unchanged, but it's more powerful than it seems!

"""
WHAT IS RunnablePassthrough?
- A component that passes input through to output WITHOUT modification
- Seems simple, but it's crucial for data flow in LCEL chains
- Used to preserve original input while transforming other parts

Think of it like a "passthrough cable" in audio equipment - data goes through unchanged
"""

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Mock LLM for demonstration (in real use, import ChatOpenAI)
class MockLLM:
    """Simulated LLM for demonstration"""
    def invoke(self, messages):
        class Response:
            content = "[LLM Response Here]"
        return Response()
    
    def __or__(self, other):
        """Support pipe operator"""
        return lambda x: other.invoke(self.invoke(x))

llm = MockLLM()

# ==================== EXAMPLE 1: BASIC PASSTHROUGH ====================
print("="*70)
print("EXAMPLE 1: BASIC RunnablePassthrough")
print("="*70 + "\n")

def example1_basic():
    """
    RunnablePassthrough simply passes the input through unchanged
    """
    
    # Create a simple passthrough
    passthrough = RunnablePassthrough()
    
    # Whatever you pass in, you get back
    input_data = {"name": "Alice", "age": 30}
    output = passthrough.invoke(input_data)
    
    print("INPUT:")
    print(input_data)
    print("\nOUTPUT (unchanged):")
    print(output)
    print("\nThey are the same:", input_data == output)
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example1_basic()

# ==================== EXAMPLE 2: PASSTHROUGH IN A CHAIN ====================
print("EXAMPLE 2: Using RunnablePassthrough in a Chain")
print("="*70 + "\n")

def example2_in_chain():
    """
    Why use passthrough in a chain? To preserve original input!
    """
    
    # Without passthrough - you lose original data
    print("WITHOUT PASSTHROUGH:")
    print("-"*70)
    
    def uppercase_transform(x):
        """Transform input to uppercase"""
        if isinstance(x, dict) and "text" in x:
            return x["text"].upper()
        return str(x).upper()
    
    chain_without = uppercase_transform
    
    input_data = {"text": "hello world", "metadata": "important info"}
    result = chain_without(input_data)
    
    print(f"Input: {input_data}")
    print(f"Output: {result}")
    print("❌ Lost metadata!\n")
    
    # With passthrough - preserve original data
    print("WITH PASSTHROUGH:")
    print("-"*70)
    
    # Using RunnablePassthrough to preserve the whole input
    chain_with = RunnablePassthrough()
    
    result = chain_with.invoke(input_data)
    
    print(f"Input: {input_data}")
    print(f"Output: {result}")
    print("✓ Metadata preserved!")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example2_in_chain()

# ==================== EXAMPLE 3: PASSTHROUGH WITH ASSIGNMENT ====================
print("EXAMPLE 3: RunnablePassthrough.assign() - The Power Move!")
print("="*70 + "\n")

def example3_assign():
    """
    RunnablePassthrough.assign() is SUPER USEFUL!
    It keeps the original input AND adds new fields
    """
    
    # Original input
    input_data = {
        "user": "Alice",
        "query": "What's the weather?"
    }
    
    print("ORIGINAL INPUT:")
    print(input_data)
    print()
    
    # Add new fields while keeping original
    chain = RunnablePassthrough.assign(
        timestamp=lambda x: "2024-01-15 10:30:00",
        query_length=lambda x: len(x["query"]),
        user_uppercase=lambda x: x["user"].upper()
    )
    
    result = chain.invoke(input_data)
    
    print("OUTPUT (with assigned fields):")
    print(result)
    print("\n✓ Original fields: 'user' and 'query' are still there")
    print("✓ New fields: 'timestamp', 'query_length', 'user_uppercase' added")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example3_assign()

# ==================== EXAMPLE 4: REAL-WORLD USE CASE ====================
print("EXAMPLE 4: Real-World Prompt Template Use Case")
print("="*70 + "\n")

def example4_real_world():
    """
    This is THE most common use case for RunnablePassthrough
    It solves the problem: "How do I pass data to my prompt?"
    """
    
    # Create a prompt that needs multiple variables
    prompt = ChatPromptTemplate.from_template(
        "Translate this {source_language} text to {target_language}: {text}"
    )
    
    # METHOD 1: Without RunnablePassthrough (manual)
    print("METHOD 1 - Manual (verbose):")
    print("-"*70)
    
    input_data = {
        "source_language": "English",
        "target_language": "Spanish", 
        "text": "Hello, how are you?"
    }
    
    # You have to manually format each variable
    formatted = prompt.invoke(input_data)
    print(f"Input: {input_data}")
    print(f"Formatted prompt ready for LLM\n")
    
    # METHOD 2: With RunnablePassthrough (elegant)
    print("METHOD 2 - With RunnablePassthrough (elegant):")
    print("-"*70)
    
    # RunnablePassthrough() passes the entire input dict to the prompt
    # The prompt automatically extracts the variables it needs
    chain = RunnablePassthrough() | prompt
    
    result = chain.invoke(input_data)
    print(f"Input: {input_data}")
    print(f"Chain handles everything automatically!")
    print("\n✓ RunnablePassthrough ensures all input data reaches the prompt")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example4_real_world()

# ==================== EXAMPLE 5: COMPLEX DATA FLOW ====================
print("EXAMPLE 5: Complex Data Flow with Multiple Passthroughs")
print("="*70 + "\n")

def example5_complex():
    """
    Using RunnablePassthrough to manage complex data transformations
    """
    
    # Scenario: You have user input and need to:
    # 1. Keep original input
    # 2. Add processed versions
    # 3. Add metadata
    
    input_data = {
        "user_query": "What's the capital of France?",
        "user_id": "12345"
    }
    
    print("INPUT:")
    print(input_data)
    print()
    
    # Build a chain that enriches the data
    chain = RunnablePassthrough.assign(
        # Add query length
        query_length=lambda x: len(x["user_query"]),
        
        # Add query in lowercase
        query_lower=lambda x: x["user_query"].lower(),
        
        # Add word count
        word_count=lambda x: len(x["user_query"].split()),
        
        # Add a flag
        is_question=lambda x: "?" in x["user_query"],
        
        # Add timestamp
        timestamp=lambda x: "2024-01-15 10:30:00"
    )
    
    result = chain.invoke(input_data)
    
    print("ENRICHED OUTPUT:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Original data preserved")
    print("✓ 5 new fields added")
    print("✓ Ready for next step in pipeline")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example5_complex()

# ==================== EXAMPLE 6: PASSTHROUGH WITH PARALLEL ====================
print("EXAMPLE 6: RunnablePassthrough with RunnableParallel")
print("="*70 + "\n")

def example6_parallel():
    """
    Combining RunnablePassthrough with RunnableParallel
    This is a VERY common pattern in production code
    """
    
    # Scenario: You want to process data in multiple ways simultaneously
    
    input_data = {
        "text": "The quick brown fox jumps over the lazy dog"
    }
    
    print("INPUT:")
    print(input_data)
    print()
    
    # Process the same input in multiple ways
    parallel_chain = RunnableParallel(
        # Keep original
        original=RunnablePassthrough(),
        
        # Transform to uppercase
        uppercase=lambda x: {"text": x["text"].upper()},
        
        # Count words
        word_count=lambda x: {"count": len(x["text"].split())},
        
        # Get first word
        first_word=lambda x: {"word": x["text"].split()[0]}
    )
    
    result = parallel_chain.invoke(input_data)
    
    print("PARALLEL PROCESSING RESULTS:")
    print("-"*70)
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print("\n✓ All processes ran in parallel")
    print("✓ Original data preserved via RunnablePassthrough")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example6_parallel()

# ==================== EXAMPLE 7: COMMON PATTERN - CONTEXT PASSING ====================
print("EXAMPLE 7: THE Context Passing Pattern (Most Important!)")
print("="*70 + "\n")

def example7_context_pattern():
    """
    This is THE MOST COMMON use of RunnablePassthrough
    Passing context through a multi-step chain
    """
    
    print("SCENARIO: Multi-step RAG (Retrieval Augmented Generation)")
    print("-"*70)
    print()
    
    # Simulated components
    def retrieve_docs(input_dict):
        """Simulate retrieving relevant documents"""
        query = input_dict["question"]
        return f"[Retrieved docs about: {query}]"
    
    def format_with_context(input_dict):
        """Format prompt with question and retrieved context"""
        return f"Context: {input_dict['context']}\n\nQuestion: {input_dict['question']}"
    
    # THE PATTERN: Use RunnablePassthrough.assign to add context
    # while keeping the original question
    
    chain = (
        # Start with: {"question": "..."}
        RunnablePassthrough.assign(
            # Add retrieved context while keeping question
            context=retrieve_docs
        )
        # Now we have: {"question": "...", "context": "..."}
        | format_with_context
        # Now we have the formatted prompt ready for LLM
    )
    
    input_data = {"question": "What is LangChain?"}
    
    print("Step-by-step execution:")
    print()
    print("1. INPUT:", input_data)
    print()
    print("2. After RunnablePassthrough.assign(context=retrieve_docs):")
    intermediate = {
        "question": input_data["question"],
        "context": "[Retrieved docs about: What is LangChain?]"
    }
    print("  ", intermediate)
    print()
    print("3. After format_with_context:")
    final = chain.invoke(input_data)
    print("  ", final)
    print()
    print("✓ Question preserved through the chain")
    print("✓ Context added at the right step")
    print("✓ Both available for final formatting")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example7_context_pattern()

# ==================== EXAMPLE 8: DEBUGGING WITH PASSTHROUGH ====================
print("EXAMPLE 8: Using RunnablePassthrough for Debugging")
print("="*70 + "\n")

def example8_debugging():
    """
    RunnablePassthrough can help you debug complex chains
    """
    
    def step1(x):
        result = {"input": x, "step1_done": True}
        print(f"  After Step 1: {result}")
        return result
    
    def step2(x):
        result = {**x, "step2_done": True}
        print(f"  After Step 2: {result}")
        return result
    
    # Chain with debugging passthroughs
    chain = (
        RunnablePassthrough()  # Checkpoint 1
        | step1
        | RunnablePassthrough()  # Checkpoint 2
        | step2
        | RunnablePassthrough()  # Checkpoint 3
    )
    
    print("DEBUGGING CHAIN EXECUTION:")
    print("-"*70)
    print()
    
    result = chain.invoke("test_input")
    
    print()
    print("FINAL RESULT:", result)
    print("\n✓ RunnablePassthrough acts as checkpoints")
    print("✓ You can see data at each step")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example8_debugging()

# ==================== EXAMPLE 9: PRACTICAL COMPLETE EXAMPLE ====================
print("EXAMPLE 9: Complete Practical Example")
print("="*70 + "\n")

def example9_complete():
    """
    A complete realistic example using RunnablePassthrough
    Building a question-answering system
    """
    
    print("BUILDING A Q&A SYSTEM WITH CONTEXT")
    print("-"*70)
    print()
    
    # Simulated functions
    def get_user_context(input_dict):
        """Simulate fetching user profile"""
        return f"User {input_dict.get('user_id', 'unknown')} - Premium Member"
    
    def retrieve_knowledge(input_dict):
        """Simulate knowledge base retrieval"""
        return f"Knowledge about: {input_dict['question']}"
    
    def calculate_priority(input_dict):
        """Calculate priority based on user and question"""
        return "HIGH" if "urgent" in input_dict['question'].lower() else "NORMAL"
    
    # Build the chain
    chain = (
        # Start with basic input
        RunnablePassthrough.assign(
            # Add user context
            user_context=get_user_context,
        )
        | RunnablePassthrough.assign(
            # Add knowledge (can access previous fields)
            knowledge=retrieve_knowledge,
        )
        | RunnablePassthrough.assign(
            # Add priority (can access all previous fields)
            priority=calculate_priority,
        )
    )
    
    # Test inputs
    inputs = [
        {"user_id": "123", "question": "How do I reset my password?"},
        {"user_id": "456", "question": "URGENT: System is down!"}
    ]
    
    for input_data in inputs:
        print(f"INPUT: {input_data}")
        result = chain.invoke(input_data)
        print(f"OUTPUT:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print()
    
    print("✓ Each step added new data")
    print("✓ Original input preserved throughout")
    print("✓ Later steps can access earlier enrichments")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example9_complete()

# ==================== SUMMARY ====================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║              RUNNABLEPASSTHROUGH - COMPLETE GUIDE                    ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT IS IT?
A component that passes input through to output unchanged. Simple concept,
but essential for building complex LCEL chains.

KEY METHODS:

1. RunnablePassthrough()
   - Passes data through unchanged
   - Usage: chain = RunnablePassthrough() | next_step

2. RunnablePassthrough.assign(**kwargs)
   - Keeps original input AND adds new fields
   - Usage: RunnablePassthrough.assign(new_field=lambda x: transform(x))
   - MOST COMMONLY USED!

WHEN TO USE IT?

✓ Preserving original input in a chain
✓ Adding new fields while keeping existing ones
✓ Passing context through multi-step pipelines
✓ Building RAG (Retrieval Augmented Generation) systems
✓ Debugging complex chains
✓ Managing data flow in LCEL

COMMON PATTERNS:

Pattern 1 - Context Passing (RAG):
    chain = (
        RunnablePassthrough.assign(context=retriever)
        | prompt
        | llm
    )

Pattern 2 - Data Enrichment:
    chain = (
        RunnablePassthrough.assign(
            field1=transform1,
            field2=transform2
        )
        | next_step
    )

Pattern 3 - Parallel + Passthrough:
    chain = RunnableParallel(
        original=RunnablePassthrough(),
        transformed=transformer
    )

WHY IS IT IMPORTANT?

Without RunnablePassthrough, you'd lose data as it flows through chains.
With it, you can:
- Keep original data accessible
- Add computed fields incrementally
- Build complex pipelines that maintain state
- Create more readable, maintainable code

REMEMBER:
Think of RunnablePassthrough as a "data highway" - it ensures your data
gets where it needs to go without getting lost along the way!
""")