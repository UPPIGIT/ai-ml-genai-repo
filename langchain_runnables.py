"""
LangChain Runnables: From Basic to Advanced with Gemini Models
============================================================

This file demonstrates LangChain Runnable patterns from simple chains to complex
multi-step workflows, including integration with Google's Gemini models.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain.schema.runnable import (
    Runnable, 
    RunnablePassthrough, 
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
    RunnableSequence
)
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import StreamingStdOutCallbackHandler

# Pydantic models
from pydantic import BaseModel, Field

# Mock LLM for demonstration (replace with actual Gemini when available)
class MockGeminiLLM:
    """Mock Gemini LLM for demonstration purposes."""
    
    def __init__(self, model_name="gemini-pro"):
        self.model_name = model_name
        self.temperature = 0.7
    
    def invoke(self, prompt: str) -> str:
        """Mock invoke method."""
        return f"Mock response to: {prompt[:50]}..."
    
    async def ainvoke(self, prompt: str) -> str:
        """Mock async invoke method."""
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Async mock response to: {prompt[:50]}..."

# =============================================================================
# BASIC RUNNABLE EXAMPLES
# =============================================================================

def basic_runnable_chain():
    """
    Simple linear chain: Prompt -> LLM -> Output Parser
    This is the most basic runnable pattern.
    """
    print("=== BASIC RUNNABLE CHAIN ===")
    
    # Initialize Gemini model (using mock for demo)
    # In real usage: llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    llm = MockGeminiLLM("gemini-pro")
    
    # Create prompt
    prompt = PromptTemplate.from_template(
        "Explain {topic} in simple terms for a {audience}."
    )
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Create the chain using the | operator (LCEL - LangChain Expression Language)
    chain = prompt | llm | output_parser
    
    # Alternative syntax for creating chains
    # chain = RunnableSequence(first=prompt, middle=[llm], last=output_parser)
    
    print("Chain structure:")
    print(f"Prompt -> {llm.model_name} -> String Parser")
    
    # Test the chain
    result = chain.invoke({
        "topic": "machine learning",
        "audience": "beginners"
    })
    
    print(f"Result: {result}")
    return chain

def runnable_with_passthrough():
    """
    Using RunnablePassthrough to preserve input data through the chain.
    Useful when you need both the original input and the LLM output.
    """
    print("\n=== RUNNABLE WITH PASSTHROUGH ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    prompt = PromptTemplate.from_template(
        "Summarize this text: {text}"
    )
    
    # Chain that preserves original input alongside the summary
    chain = {
        "original_text": RunnablePassthrough(),  # Pass through the input
        "summary": prompt | llm | StrOutputParser()  # Generate summary
    }
    
    result = chain.invoke({"text": "Long article about artificial intelligence..."})
    print("Result structure preserves both input and output:")
    print(result)
    
    return chain

def runnable_lambda_examples():
    """
    Using RunnableLambda for custom processing steps.
    Allows insertion of custom Python functions into chains.
    """
    print("\n=== RUNNABLE LAMBDA EXAMPLES ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    # Custom functions as runnables
    def extract_keywords(text: str) -> List[str]:
        """Extract simple keywords from text."""
        # Simple keyword extraction (in practice, use NLP libraries)
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 4]
        return keywords[:5]  # Return top 5
    
    def count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def format_analysis(data: Dict) -> str:
        """Format the analysis results."""
        return f"""
Text Analysis Results:
- Word count: {data['word_count']}
- Keywords: {', '.join(data['keywords'])}
- AI Summary: {data['summary']}
"""
    
    # Create chain with lambda functions
    analysis_chain = {
        "text": RunnablePassthrough(),
        "word_count": RunnableLambda(count_words),
        "keywords": RunnableLambda(extract_keywords),
        "summary": PromptTemplate.from_template("Summarize in one sentence: {text}") | llm | StrOutputParser()
    } | RunnableLambda(format_analysis)
    
    # Test the chain
    sample_text = "Artificial intelligence is transforming various industries through automation and intelligent decision making."
    result = analysis_chain.invoke(sample_text)
    print("Analysis result:")
    print(result)
    
    return analysis_chain

# =============================================================================
# PARALLEL RUNNABLE EXAMPLES
# =============================================================================

def parallel_processing_example():
    """
    Using RunnableParallel to execute multiple operations simultaneously.
    Great for performance when operations don't depend on each other.
    """
    print("\n=== PARALLEL PROCESSING EXAMPLE ===")
    
    # Different Gemini models for different tasks
    creative_llm = MockGeminiLLM("gemini-pro")
    analytical_llm = MockGeminiLLM("gemini-pro")
    
    # Define parallel operations
    parallel_chain = RunnableParallel(
        # Creative writing task
        creative_response=PromptTemplate.from_template(
            "Write a creative story opening about: {topic}"
        ) | creative_llm | StrOutputParser(),
        
        # Analytical task
        analytical_response=PromptTemplate.from_template(
            "Provide a technical analysis of: {topic}"
        ) | analytical_llm | StrOutputParser(),
        
        # Simple processing task
        word_count=RunnableLambda(lambda x: len(x["topic"].split())),
        
        # Metadata
        timestamp=RunnableLambda(lambda x: datetime.now().isoformat())
    )
    
    # Execute all operations in parallel
    result = parallel_chain.invoke({"topic": "quantum computing applications"})
    
    print("Parallel execution results:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    return parallel_chain

async def async_parallel_example():
    """
    Asynchronous parallel processing for better performance.
    """
    print("\n=== ASYNC PARALLEL EXAMPLE ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    # Create async chain
    async_chain = RunnableParallel(
        task1=PromptTemplate.from_template("Explain {topic} for experts") | RunnableLambda(llm.ainvoke),
        task2=PromptTemplate.from_template("Explain {topic} for beginners") | RunnableLambda(llm.ainvoke),
        task3=PromptTemplate.from_template("List pros and cons of {topic}") | RunnableLambda(llm.ainvoke)
    )
    
    # Execute asynchronously
    start_time = datetime.now()
    result = await async_chain.ainvoke({"topic": "blockchain technology"})
    end_time = datetime.now()
    
    print(f"Async execution completed in: {end_time - start_time}")
    print("Results:", list(result.keys()))
    
    return result

# =============================================================================
# CONDITIONAL BRANCHING
# =============================================================================

def conditional_branching_example():
    """
    Using RunnableBranch for conditional logic in chains.
    Routes execution based on input conditions.
    """
    print("\n=== CONDITIONAL BRANCHING EXAMPLE ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    # Define different prompt templates for different content types
    technical_prompt = PromptTemplate.from_template(
        "Provide a detailed technical explanation of {content} including implementation details."
    )
    
    creative_prompt = PromptTemplate.from_template(
        "Write a creative and engaging explanation of {content} using analogies and stories."
    )
    
    business_prompt = PromptTemplate.from_template(
        "Explain {content} from a business perspective, focusing on ROI and practical applications."
    )
    
    # Create branching logic
    branching_chain = RunnableBranch(
        # Condition 1: Technical audience
        (
            lambda x: x.get("audience") == "technical",
            technical_prompt | llm | StrOutputParser()
        ),
        # Condition 2: Creative audience
        (
            lambda x: x.get("audience") == "creative",
            creative_prompt | llm | StrOutputParser()
        ),
        # Condition 3: Business audience
        (
            lambda x: x.get("audience") == "business",
            business_prompt | llm | StrOutputParser()
        ),
        # Default case
        PromptTemplate.from_template("Provide a general explanation of {content}") | llm | StrOutputParser()
    )
    
    # Test different conditions
    test_cases = [
        {"content": "machine learning", "audience": "technical"},
        {"content": "machine learning", "audience": "creative"},
        {"content": "machine learning", "audience": "business"},
        {"content": "machine learning", "audience": "unknown"}  # Default case
    ]
    
    for i, test_case in enumerate(test_cases):
        result = branching_chain.invoke(test_case)
        print(f"Test {i+1} ({test_case['audience']}): {result[:50]}...")
    
    return branching_chain

# =============================================================================
# ADVANCED PATTERNS
# =============================================================================

class ResearchResult(BaseModel):
    """Model for structured research output."""
    topic: str = Field(description="Research topic")
    key_findings: List[str] = Field(description="Main research findings")
    methodology: str = Field(description="Research methodology used")
    confidence_score: float = Field(description="Confidence in findings (0-1)")
    sources: List[str] = Field(description="Information sources")
    recommendations: List[str] = Field(description="Actionable recommendations")

def advanced_research_chain():
    """
    Advanced chain combining multiple patterns for comprehensive research workflow.
    Demonstrates complex multi-step reasoning with structured output.
    """
    print("\n=== ADVANCED RESEARCH CHAIN ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    # Step 1: Research planning
    planning_prompt = PromptTemplate.from_template("""
    Create a research plan for: {topic}
    
    Consider:
    - Key questions to investigate
    - Potential information sources
    - Research methodology
    - Expected challenges
    
    Research Plan:
    """)
    
    # Step 2: Information gathering (simulated)
    def simulate_research(plan: str) -> Dict[str, Any]:
        """Simulate gathering research information."""
        return {
            "plan": plan,
            "findings": [
                "Finding 1: Key insight about the topic",
                "Finding 2: Important trend or pattern",
                "Finding 3: Critical consideration or limitation"
            ],
            "sources": ["Source A", "Source B", "Source C"],
            "raw_data": "Simulated research data..."
        }
    
    # Step 3: Analysis and synthesis
    analysis_prompt = PromptTemplate.from_template("""
    Based on the research findings below, provide a comprehensive analysis.
    
    Research Plan: {plan}
    Key Findings: {findings}
    
    Please analyze:
    1. What are the most important insights?
    2. What patterns or trends emerge?
    3. What are the implications?
    4. What recommendations can you make?
    
    Analysis:
    """)
    
    # Step 4: Structured output generation
    output_parser = PydanticOutputParser(pydantic_object=ResearchResult)
    
    final_prompt = PromptTemplate.from_template("""
    Convert the following research analysis into a structured format:
    
    Topic: {topic}
    Analysis: {analysis}
    Sources: {sources}
    
    {format_instructions}
    """)
    
    # Create the complete research chain
    research_chain = (
        {
            "topic": RunnablePassthrough(),
            "plan": planning_prompt | llm | StrOutputParser()
        }
        | RunnableLambda(lambda x: {**x, **simulate_research(x["plan"])})
        | {
            "topic": lambda x: x["topic"],
            "sources": lambda x: x["sources"],
            "analysis": analysis_prompt | llm | StrOutputParser()
        }
        | final_prompt.partial(format_instructions=output_parser.get_format_instructions())
        | llm
        | StrOutputParser()
    )
    
    # Test the research chain
    topic = "impact of artificial intelligence on healthcare"
    try:
        result = research_chain.invoke(topic)
        print(f"Research completed for: {topic}")
        print(f"Result preview: {result[:100]}...")
    except Exception as e:
        print(f"Research chain error: {e}")
    
    return research_chain

# =============================================================================
# STREAMING AND CALLBACKS
# =============================================================================

class CustomCallback:
    """Custom callback for monitoring chain execution."""
    
    def __init__(self):
        self.steps = []
        self.start_time = None
    
    def on_chain_start(self, inputs):
        self.start_time = datetime.now()
        self.steps.append(f"Chain started with inputs: {inputs}")
    
    def on_chain_end(self, outputs):
        duration = datetime.now() - self.start_time
        self.steps.append(f"Chain completed in {duration.total_seconds():.2f}s")
    
    def get_summary(self):
        return "\n".join(self.steps)

def streaming_example():
    """
    Example of streaming responses and custom callbacks.
    """
    print("\n=== STREAMING EXAMPLE ===")
    
    # Note: Actual streaming would require real Gemini integration
    llm = MockGeminiLLM("gemini-pro")
    
    prompt = PromptTemplate.from_template(
        "Write a detailed explanation of {topic}. Make it comprehensive and informative."
    )
    
    # Create streaming chain (simplified for demo)
    streaming_chain = prompt | llm | StrOutputParser()
    
    # Simulate streaming with callback
    callback = CustomCallback()
    
    print("Simulating streaming response...")
    callback.on_chain_start({"topic": "quantum computing"})
    
    # In real implementation, this would stream tokens
    result = streaming_chain.invoke({"topic": "quantum computing"})
    
    callback.on_chain_end({"result": result})
    
    print("Streaming completed!")
    print("Callback summary:")
    print(callback.get_summary())
    
    return streaming_chain

# =============================================================================
# MEMORY AND STATE MANAGEMENT
# =============================================================================

class ConversationMemory:
    """Simple conversation memory for maintaining context."""
    
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add_exchange(self, human_input: str, ai_response: str):
        """Add a conversation exchange to memory."""
        self.history.append({
            "human": human_input,
            "ai": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """Get conversation context as string."""
        if not self.history:
            return "No previous conversation."
        
        context = "Previous conversation:\n"
        for exchange in self.history:
            context += f"Human: {exchange['human']}\n"
            context += f"AI: {exchange['ai']}\n\n"
        
        return context

def stateful_conversation_chain():
    """
    Advanced chain with memory and state management.
    Maintains conversation context across multiple interactions.
    """
    print("\n=== STATEFUL CONVERSATION CHAIN ===")
    
    llm = MockGeminiLLM("gemini-pro")
    memory = ConversationMemory()
    
    conversation_prompt = PromptTemplate.from_template("""
    {context}
    
    Current question: {question}
    
    Please provide a helpful response that takes into account the conversation history.
    Be consistent with previous responses and build upon the established context.
    
    Response:
    """)
    
    def add_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Add conversation context to inputs."""
        return {
            **inputs,
            "context": memory.get_context()
        }
    
    def save_to_memory(result: str, original_input: Dict[str, Any]) -> str:
        """Save the exchange to memory."""
        memory.add_exchange(original_input["question"], result)
        return result
    
    # Create stateful chain
    stateful_chain = (
        RunnableLambda(add_context)
        | conversation_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda result: save_to_memory(result, {"question": "stored_question"}))
    )
    
    # Simulate conversation
    questions = [
        "What is machine learning?",
        "Can you give me an example?",
        "How does this relate to what we discussed earlier?",
        "What are the practical applications?"
    ]
    
    print("Simulating conversation:")
    for i, question in enumerate(questions):
        print(f"\nTurn {i+1}:")
        print(f"Human: {question}")
        
        # Note: In real implementation, we'd need to pass the question properly
        result = f"Mock response {i+1} considering context: {question}"
        memory.add_exchange(question, result)
        
        print(f"AI: {result}")
    
    print(f"\nConversation history length: {len(memory.history)}")
    
    return stateful_chain, memory

# =============================================================================
# ERROR HANDLING AND RETRY LOGIC
# =============================================================================

class RetryableRunnable(Runnable):
    """Custom runnable with retry logic and error handling."""
    
    def __init__(self, runnable: Runnable, max_retries: int = 3, delay: float = 1.0):
        self.runnable = runnable
        self.max_retries = max_retries
        self.delay = delay
    
    def invoke(self, input: Any, config: Optional[Dict] = None) -> Any:
        """Invoke with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self.runnable.invoke(input, config)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.delay}s...")
                    asyncio.sleep(self.delay)
                else:
                    print(f"All {self.max_retries + 1} attempts failed.")
        
        raise last_error

def error_handling_example():
    """
    Example of robust error handling and retry logic in chains.
    """
    print("\n=== ERROR HANDLING EXAMPLE ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    # Create a potentially failing chain
    def potentially_failing_function(x):
        import random
        if random.random() < 0.7:  # 70% chance of failure for demo
            raise Exception("Simulated API failure")
        return f"Success: {x}"
    
    # Create chain with error handling
    base_chain = (
        PromptTemplate.from_template("Process this: {input}")
        | llm
        | StrOutputParser()
        | RunnableLambda(potentially_failing_function)
    )
    
    # Wrap with retry logic
    robust_chain = RetryableRunnable(base_chain, max_retries=3, delay=0.1)
    
    # Test error handling
    try:
        result = robust_chain.invoke({"input": "test data"})
        print(f"Chain succeeded: {result}")
    except Exception as e:
        print(f"Chain failed after all retries: {e}")
    
    return robust_chain

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

def performance_optimization_example():
    """
    Examples of performance optimization techniques for runnables.
    """
    print("\n=== PERFORMANCE OPTIMIZATION EXAMPLE ===")
    
    llm = MockGeminiLLM("gemini-pro")
    
    # Technique 1: Batch processing
    def batch_process_texts(texts: List[str]) -> List[str]:
        """Process multiple texts in a single batch."""
        return [f"Processed: {text}" for text in texts]
    
    batch_chain = RunnableLambda(batch_process_texts)
    
    # Technique 2: Caching results
    cache = {}
    
    def cached_llm_call(prompt: str) -> str:
        """LLM call with simple caching."""
        if prompt in cache:
            print(f"Cache hit for: {prompt[:30]}...")
            return cache[prompt]
        
        result = llm.invoke(prompt)
        cache[prompt] = result
        print(f"Cache miss, stored result for: {prompt[:30]}...")
        return result
    
    cached_chain = RunnableLambda(cached_llm_call)
    
    # Technique 3: Parallel batch processing
    def parallel_batch_processing(inputs: List[Dict]) -> List[str]:
        """Process multiple inputs in parallel."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(llm.invoke, inp["text"]) for inp in inputs]
            results = [future.result() for future in futures]
        return results
    
    parallel_chain = RunnableLambda(parallel_batch_processing)
    
    # Test performance techniques
    print("Testing batch processing:")
    batch_result = batch_chain.invoke(["text1", "text2", "text3"])
    print(f"Batch result: {batch_result}")
    
    print("\nTesting caching:")
    cached_chain.invoke("What is AI?")  # Cache miss
    cached_chain.invoke("What is AI?")  # Cache hit
    
    print(f"\nCache size: {len(cache)}")
    
    return {
        "batch_chain": batch_chain,
        "cached_chain": cached_chain,
        "parallel_chain": parallel_chain
    }

# =============================================================================
# INTEGRATION WITH GEMINI MODELS
# =============================================================================

def gemini_integration_example():
    """
    Example of integrating with actual Gemini models.
    Note: Requires proper API setup and credentials.
    """
    print("\n=== GEMINI INTEGRATION EXAMPLE ===")
    
    # In real usage, uncomment and configure:
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # 
    # # Initialize Gemini models
    # gemini_pro = ChatGoogleGenerativeAI(
    #     model="gemini-pro",
    #     temperature=0.7,
    #     convert_system_message_to_human=True
    # )
    # 
    # gemini_pro_vision = ChatGoogleGenerativeAI(
    #     model="gemini-pro-vision",
    #     temperature=0.3
    # )
    
    # For demo, using mock models
    gemini_pro = MockGeminiLLM("gemini-pro")
    gemini_pro_vision = MockGeminiLLM("gemini-pro-vision")
    
    # Multi-modal chain for text and vision
    multi_modal_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with vision capabilities."),
        ("human", "Analyze this content: {content}")
    ])
    
    # Create specialized chains for different Gemini capabilities
    text_analysis_chain = (
        PromptTemplate.from_template("Provide detailed analysis of: {text}")
        | gemini_pro
        | StrOutputParser()
    )
    
    # Simulated vision chain (would work with actual images)
    vision_analysis_chain = (
        PromptTemplate.from_template("Describe what you see in the image: {image_description}")
        | gemini_pro_vision
        | StrOutputParser()
    )
    
    # Combined workflow
    combined_analysis = RunnableParallel(
        text_analysis=text_analysis_chain,
        vision_analysis=vision_analysis_chain,
        metadata=RunnableLambda(lambda x: {
            "timestamp": datetime.now().isoformat(),
            "model_versions": {
                "text": "gemini-pro",
                "vision": "gemini-pro-vision"
            }
        })
    )
    
    # Test the integration
    test_input = {
        "text": "Artificial intelligence is revolutionizing healthcare",
        "image_description": "A medical AI system analyzing patient data"
    }
    
    result = combined_analysis.invoke(test_input)
    print("Gemini integration results:")
    for key, value in result.items():
        print(f"{key}: {str(value)[:50]}...")
    
    return combined_analysis

# =============================================================================
# DEMONSTRATION RUNNER
# =============================================================================

def demonstrate_all_runnables():
    """
    Run all runnable examples to demonstrate different patterns.
    """
    print("=" * 80)
    print("LANGCHAIN RUNNABLES: BASIC TO ADVANCED EXAMPLES")
    print("=" * 80)
    
    examples = []
    
    # Basic examples
    examples.append(("Basic Chain", basic_runnable_chain))
    examples.append(("Passthrough", runnable_with_passthrough))
    examples.append(("Lambda Functions", runnable_lambda_examples))
    
    # Parallel processing
    examples.append(("Parallel Processing", parallel_processing_example))
    
    # Conditional logic
    examples.append(("Conditional Branching", conditional_branching_example))
    
    # Advanced patterns
    examples.append(("Advanced Research", advanced_research_chain))
    examples.append(("Streaming", streaming_example))
    examples.append(("Stateful Conversation", lambda: stateful_conversation_chain()[0]))
    examples.append(("Error Handling", error_handling_example))
    examples.append(("Performance Optimization", performance_optimization_example))
    examples.append(("Gemini Integration", gemini_integration_example))
    
    # Run all examples
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("\n" + "=" * 80)
    print("ALL RUNNABLE EXAMPLES COMPLETED!")
    print("=" * 80)

async def demonstrate_async_examples():
    """
    Run async examples separately.
    """
    print("\n=== ASYNC EXAMPLES ===")
    try:
        await async_parallel_example()
        print("✅ Async examples completed")
    except Exception as e:
        print(f"❌ Async examples failed: {e}")

# Usage examples and best practices
RUNNABLE_BEST_PRACTICES = """
LANGCHAIN RUNNABLE BEST PRACTICES:

1. COMPOSITION PATTERNS:
   - Use | operator for linear chains
   - Use RunnableParallel for independent operations
   - Use RunnableBranch for conditional logic

2. PERFORMANCE:
   - Leverage parallel execution when possible
   - Implement caching for expensive operations
   - Use async patterns for I/O-bound operations

3. ERROR HANDLING:
   - Wrap chains in retry logic
   - Implement graceful degradation
   - Log errors for debugging

4. MEMORY MANAGEMENT:
   - Implement conversation memory for chatbots
   - Clear cache periodically to prevent memory leaks
   - Use streaming for large responses

5. TESTING:
   - Mock LLM calls for unit tests
   - Test error conditions
   - Validate output formats

6. MONITORING:
   - Use callbacks for observability
   - Track performance metrics
   - Monitor API usage and costs
"""

if __name__ == "__main__":
    # Run synchronous examples
    demonstrate_all_runnables()
    
    # Run async examples
    asyncio.run(demonstrate_async_examples())
    
    # Print best practices
    print(RUNNABLE_BEST_PRACTICES)
