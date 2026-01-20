# 11_calling_llms_with_prompts.py
# Complete examples of calling actual LLMs with LangChain prompts
# This shows real integration with OpenAI, Anthropic (Claude), and other providers

"""
INSTALLATION:
pip install langchain langchain-openai langchain-anthropic langchain-google-genai python-dotenv

SETUP:
Create a .env file with your API keys:
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
"""

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# ==================== EXAMPLE 1: BASIC LLM CALL WITH OPENAI ====================
print("="*70)
print("EXAMPLE 1: BASIC OpenAI LLM CALL")
print("="*70 + "\n")

def example1_basic_openai():
    """Simple prompt with OpenAI"""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
        max_tokens=500    # Maximum length of response
    )
    
    # Create a simple prompt
    prompt = PromptTemplate(
        template="Write a {length} paragraph about {topic}.",
        input_variables=["length", "topic"]
    )
    
    # Format the prompt
    formatted_prompt = prompt.format(length="short", topic="artificial intelligence")
    
    print("PROMPT SENT:")
    print(formatted_prompt)
    print("\n" + "-"*70 + "\n")
    
    # Call the LLM
    response = llm.invoke(formatted_prompt)
    
    print("RESPONSE:")
    print(response.content)
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example1_basic_openai()

# ==================== EXAMPLE 2: CHAT MODEL WITH CLAUDE ====================
print("EXAMPLE 2: CHAT MODEL WITH ANTHROPIC CLAUDE")
print("="*70 + "\n")

def example2_claude_chat():
    """Chat-based interaction with Claude"""
    
    # Initialize Claude
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.5,
        max_tokens=1000
    )
    
    # Create a chat prompt with system and user messages
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding assistant. Provide clear, concise answers with code examples."),
        ("human", "Explain {concept} in {language} with an example.")
    ])
    
    # Format the messages
    messages = chat_prompt.format_messages(
        concept="list comprehension",
        language="Python"
    )
    
    print("MESSAGES SENT:")
    for msg in messages:
        print(f"{msg.type.upper()}: {msg.content}")
    print("\n" + "-"*70 + "\n")
    
    # Call Claude
    response = llm.invoke(messages)
    
    print("CLAUDE'S RESPONSE:")
    print(response.content)
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example2_claude_chat()

# ==================== EXAMPLE 3: USING CHAINS FOR REPEATED CALLS ====================
print("EXAMPLE 3: USING LANGCHAIN CHAINS")
print("="*70 + "\n")

def example3_langchain_chain():
    """Use LLMChain for cleaner code and reusability"""
    
    from langchain.chains import LLMChain
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
    
    # Create prompt
    prompt = PromptTemplate(
        template="Generate a creative {item} name for a {business_type} business.",
        input_variables=["item", "business_type"]
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain multiple times with different inputs
    inputs = [
        {"item": "product", "business_type": "coffee shop"},
        {"item": "service", "business_type": "fitness center"},
        {"item": "app", "business_type": "food delivery"}
    ]
    
    for inp in inputs:
        print(f"INPUT: {inp}")
        result = chain.run(**inp)
        print(f"OUTPUT: {result}\n")
        print("-"*70 + "\n")

# Uncomment to run:
# example3_langchain_chain()

# ==================== EXAMPLE 4: FEW-SHOT WITH LLM ====================
print("EXAMPLE 4: FEW-SHOT LEARNING WITH LLM")
print("="*70 + "\n")

def example4_few_shot_with_llm():
    """Use few-shot prompts to guide LLM behavior"""
    
    from langchain.chains import LLMChain
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    # Create examples for sentiment analysis
    examples = [
        {"text": "I absolutely love this product!", "sentiment": "Positive"},
        {"text": "This is the worst experience ever.", "sentiment": "Negative"},
        {"text": "It's okay, nothing special.", "sentiment": "Neutral"},
    ]
    
    example_formatter = PromptTemplate(
        input_variables=["text", "sentiment"],
        template="Text: {text}\nSentiment: {sentiment}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_formatter,
        prefix="Analyze the sentiment of the following texts:\n",
        suffix="Text: {input}\nSentiment:",
        input_variables=["input"]
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=few_shot_prompt)
    
    # Test with new inputs
    test_texts = [
        "The service was decent but could be better.",
        "Amazing quality! Highly recommend!",
        "I'm disappointed with the results."
    ]
    
    for text in test_texts:
        result = chain.run(input=text)
        print(f"TEXT: {text}")
        print(f"SENTIMENT: {result.strip()}\n")
        print("-"*70 + "\n")

# Uncomment to run:
# example4_few_shot_with_llm()

# ==================== EXAMPLE 5: STREAMING RESPONSES ====================
print("EXAMPLE 5: STREAMING RESPONSES (Real-time)")
print("="*70 + "\n")

def example5_streaming():
    """Stream responses token by token for better UX"""
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True  # Enable streaming
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative storyteller."),
        ("human", "Write a very short story about {topic}.")
    ])
    
    messages = prompt.format_messages(topic="a time-traveling cat")
    
    print("STREAMING RESPONSE:")
    print("-"*70)
    
    # Stream the response
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-"*70 + "\n")
    print("="*70 + "\n")

# Uncomment to run:
# example5_streaming()

# ==================== EXAMPLE 6: BATCH PROCESSING ====================
print("EXAMPLE 6: BATCH PROCESSING MULTIPLE INPUTS")
print("="*70 + "\n")

def example6_batch_processing():
    """Process multiple inputs efficiently"""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    prompt = PromptTemplate(
        template="Summarize this in one sentence: {text}",
        input_variables=["text"]
    )
    
    # Prepare multiple inputs
    texts = [
        "Artificial intelligence is transforming how we work and live. From healthcare to transportation, AI is making significant impacts.",
        "Climate change poses serious threats to our planet. Rising temperatures and extreme weather events are becoming more common.",
        "The space exploration industry is experiencing rapid growth. Private companies are now launching satellites and planning missions to Mars."
    ]
    
    # Format prompts for batch
    formatted_prompts = [prompt.format(text=text) for text in texts]
    
    # Batch process
    responses = llm.batch(formatted_prompts)
    
    print("BATCH PROCESSING RESULTS:")
    for i, (original, response) in enumerate(zip(texts, responses), 1):
        print(f"\nINPUT {i}: {original[:50]}...")
        print(f"SUMMARY: {response.content}")
        print("-"*70)

# Uncomment to run:
# example6_batch_processing()

# ==================== EXAMPLE 7: WITH OUTPUT PARSER ====================
print("EXAMPLE 7: STRUCTURED OUTPUT WITH PARSER")
print("="*70 + "\n")

def example7_with_output_parser():
    """Get structured JSON output from LLM"""
    
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import List
    
    # Define structure
    class MovieReview(BaseModel):
        title: str = Field(description="Movie title")
        rating: int = Field(description="Rating from 1-10")
        pros: List[str] = Field(description="Positive aspects")
        cons: List[str] = Field(description="Negative aspects")
        recommendation: str = Field(description="Watch or skip")
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=MovieReview)
    
    # Create prompt with format instructions
    prompt = PromptTemplate(
        template="Review this movie: {movie_name}\n{format_instructions}",
        input_variables=["movie_name"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    # Create chain
    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    
    # Get structured output
    result = chain.run(movie_name="The Matrix")
    
    print("STRUCTURED OUTPUT:")
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}/10")
    print(f"Pros: {', '.join(result.pros)}")
    print(f"Cons: {', '.join(result.cons)}")
    print(f"Recommendation: {result.recommendation}")
    print("\n" + "="*70 + "\n")

# Uncomment to run:
# example7_with_output_parser()

# ==================== EXAMPLE 8: CONVERSATION WITH MEMORY ====================
print("EXAMPLE 8: CONVERSATION WITH MEMORY")
print("="*70 + "\n")

def example8_conversation_memory():
    """Maintain conversation context across multiple exchanges"""
    
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create memory
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # Shows the full prompt being sent
    )
    
    # Have a conversation
    exchanges = [
        "Hi, my name is Alice and I'm learning Python.",
        "What are some good resources for beginners?",
        "What was my name again?"  # Tests if AI remembers
    ]
    
    print("CONVERSATION:")
    print("-"*70)
    for user_input in exchanges:
        print(f"\nYOU: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI: {response}")
        print("-"*70)
    
    # Show conversation history
    print("\nCONVERSATION HISTORY:")
    print(memory.buffer)

# Uncomment to run:
# example8_conversation_memory()

# ==================== EXAMPLE 9: SEQUENTIAL CHAIN ====================
print("EXAMPLE 9: SEQUENTIAL CHAIN (Multi-Step)")
print("="*70 + "\n")

def example9_sequential_chain():
    """Chain multiple LLM calls together"""
    
    from langchain.chains import LLMChain, SequentialChain
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
    
    # Step 1: Generate a topic
    topic_prompt = PromptTemplate(
        template="Suggest an interesting topic about {subject}",
        input_variables=["subject"]
    )
    topic_chain = LLMChain(
        llm=llm,
        prompt=topic_prompt,
        output_key="topic"
    )
    
    # Step 2: Write about the topic
    writing_prompt = PromptTemplate(
        template="Write a brief paragraph about: {topic}",
        input_variables=["topic"]
    )
    writing_chain = LLMChain(
        llm=llm,
        prompt=writing_prompt,
        output_key="paragraph"
    )
    
    # Step 3: Create a title
    title_prompt = PromptTemplate(
        template="Create a catchy title for this paragraph:\n{paragraph}",
        input_variables=["paragraph"]
    )
    title_chain = LLMChain(
        llm=llm,
        prompt=title_prompt,
        output_key="title"
    )
    
    # Combine into sequential chain
    overall_chain = SequentialChain(
        chains=[topic_chain, writing_chain, title_chain],
        input_variables=["subject"],
        output_variables=["topic", "paragraph", "title"],
        verbose=True
    )
    
    # Run the chain
    result = overall_chain({"subject": "space exploration"})
    
    print("\nFINAL RESULT:")
    print(f"Title: {result['title']}")
    print(f"Topic: {result['topic']}")
    print(f"Content: {result['paragraph']}")

# Uncomment to run:
# example9_sequential_chain()

# ==================== EXAMPLE 10: ERROR HANDLING ====================
print("EXAMPLE 10: ERROR HANDLING AND RETRY")
print("="*70 + "\n")

def example10_error_handling():
    """Proper error handling when calling LLMs"""
    
    from langchain.chains import LLMChain
    import time
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = PromptTemplate(
        template="Explain {concept} simply",
        input_variables=["concept"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    def call_llm_with_retry(concept, max_retries=3):
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}...")
                result = chain.run(concept=concept)
                print("SUCCESS!")
                return result
            
            except Exception as e:
                print(f"Error: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Giving up.")
                    return None
    
    # Test the function
    result = call_llm_with_retry("quantum computing")
    if result:
        print(f"\nRESULT: {result}")

# Uncomment to run:
# example10_error_handling()

# ==================== RUNNING ALL EXAMPLES ====================
print("\n" + "="*70)
print("HOW TO RUN THESE EXAMPLES")
print("="*70)
print("""
1. Install dependencies:
   pip install langchain langchain-openai langchain-anthropic python-dotenv

2. Create .env file with your API keys:
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...

3. Uncomment the example functions above and run:
   python 11_calling_llms_with_prompts.py

Each example demonstrates a different pattern:
- Example 1: Basic OpenAI call
- Example 2: Claude chat
- Example 3: Reusable chains
- Example 4: Few-shot learning
- Example 5: Streaming responses
- Example 6: Batch processing
- Example 7: Structured output
- Example 8: Conversation memory
- Example 9: Sequential chains
- Example 10: Error handling

TIP: Start with Example 1 or 2, then progress to more advanced examples!
""")