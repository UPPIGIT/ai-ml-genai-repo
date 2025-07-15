# RetrievalQA Explained Simply with Examples

"""
RetrievalQA is like having a smart assistant that can:
1. Search through your documents to find relevant information
2. Use that information to answer your questions accurately

Think of it as: Question → Search Documents → Generate Answer
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# =============================================================================
# WHAT IS RETRIEVALQA?
# =============================================================================

"""
RetrievalQA is a chain in LangChain that combines:

1. RETRIEVAL: Finding relevant documents from your knowledge base
2. QA (Question Answering): Using those documents to answer questions

The process works like this:
User Question → Retrieve Relevant Docs → Generate Answer Based on Docs
"""

# =============================================================================
# EXAMPLE 1: Basic RetrievalQA - The Simplest Example
# =============================================================================

def basic_retrievalqa_example():
    """
    The simplest possible RetrievalQA example.
    Like asking a librarian to find books and then answer your question.
    """
    print("=== EXAMPLE 1: Basic RetrievalQA ===")
    
    # Step 1: Create some documents (your knowledge base)
    documents = [
        Document(page_content="Dogs are loyal pets that love to play fetch and go for walks."),
        Document(page_content="Cats are independent pets that like to sleep and hunt mice."),
        Document(page_content="Fish are quiet pets that live in aquariums and need daily feeding."),
        Document(page_content="Birds are social pets that can learn to talk and sing songs.")
    ]
    
    # Step 2: Create a vector store (like creating a searchable index)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Step 3: Create a retriever (the "searcher")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Get top 2 relevant docs
    
    # Step 4: Create the language model (the "answerer")
    llm = ChatOpenAI(temperature=0)
    
    # Step 5: Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means put all retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True  # Show which documents were used
    )
    
    # Step 6: Ask questions
    questions = [
        "What pets are good for playing?",
        "Which pets are independent?",
        "What do fish need?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")
        print(f"Used {len(result['source_documents'])} documents")

# =============================================================================
# EXAMPLE 2: RetrievalQA with Custom Prompt
# =============================================================================

def custom_prompt_retrievalqa():
    """
    Using a custom prompt to control how the AI answers questions.
    Like giving specific instructions to your assistant.
    """
    print("\n=== EXAMPLE 2: RetrievalQA with Custom Prompt ===")
    
    # Create documents about cooking
    cooking_docs = [
        Document(page_content="To make pasta, boil water, add salt, cook pasta for 8-12 minutes."),
        Document(page_content="Pizza dough needs flour, water, yeast, and salt. Let it rise for 1 hour."),
        Document(page_content="Scrambled eggs: crack eggs, add milk, whisk, cook on medium heat while stirring."),
        Document(page_content="Grilled cheese: butter bread, add cheese, grill until golden brown on both sides.")
    ]
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(cooking_docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Create custom prompt template
    custom_prompt = PromptTemplate(
        template="""
        You are a helpful cooking assistant. Use the following recipe information to answer the question.
        
        Recipe Information:
        {context}
        
        Question: {question}
        
        Answer the question in a friendly, step-by-step manner. If you don't know the answer from the recipes provided, say so.
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA with custom prompt
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    
    # Ask cooking questions
    cooking_questions = [
        "How do I make scrambled eggs?",
        "What ingredients do I need for pizza?",
        "How long should I cook pasta?"
    ]
    
    for question in cooking_questions:
        print(f"\nQuestion: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")

# =============================================================================
# EXAMPLE 3: Different Chain Types Explained
# =============================================================================

def chain_types_comparison():
    """
    RetrievalQA has different chain types. Let's see how they work.
    """
    print("\n=== EXAMPLE 3: Different Chain Types ===")
    
    # Create documents about space
    space_docs = [
        Document(page_content="The Sun is a star at the center of our solar system."),
        Document(page_content="Earth is the third planet from the Sun and has one moon."),
        Document(page_content="Mars is known as the Red Planet and has two small moons."),
        Document(page_content="Jupiter is the largest planet and has over 70 moons."),
        Document(page_content="Saturn is famous for its beautiful rings made of ice and rock.")
    ]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(space_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0)
    
    # Different chain types
    chain_types = ["stuff", "map_reduce", "refine"]
    
    question = "Tell me about planets and their moons."
    
    for chain_type in chain_types:
        print(f"\n--- Chain Type: {chain_type} ---")
        
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": question})
            print(f"Answer: {result['result']}")
            
            # Explain what each chain type does
            if chain_type == "stuff":
                print("📝 'stuff' puts all documents into one prompt (simplest)")
            elif chain_type == "map_reduce":
                print("🗺️ 'map_reduce' processes each document separately, then combines")
            elif chain_type == "refine":
                print("🔄 'refine' builds answer iteratively, refining with each document")
                
        except Exception as e:
            print(f"Error with {chain_type}: {e}")

# =============================================================================
# EXAMPLE 4: RetrievalQA with Filtering
# =============================================================================

def retrievalqa_with_filtering():
    """
    Sometimes you want to search only specific types of documents.
    Like asking a librarian to only look in the science section.
    """
    print("\n=== EXAMPLE 4: RetrievalQA with Filtering ===")
    
    # Create documents with metadata (categories)
    documents = [
        Document(page_content="Python is a programming language great for beginners.", 
                metadata={"category": "programming", "level": "beginner"}),
        Document(page_content="JavaScript runs in web browsers and on servers.", 
                metadata={"category": "programming", "level": "intermediate"}),
        Document(page_content="Machine learning helps computers learn from data.", 
                metadata={"category": "ai", "level": "advanced"}),
        Document(page_content="HTML is used to create web page structure.", 
                metadata={"category": "web", "level": "beginner"}),
        Document(page_content="CSS makes web pages look beautiful with styling.", 
                metadata={"category": "web", "level": "beginner"})
    ]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Create retriever with filtering
    def create_filtered_retriever(category_filter=None, level_filter=None):
        search_kwargs = {"k": 3}
        
        # Add filters if provided
        if category_filter or level_filter:
            filter_dict = {}
            if category_filter:
                filter_dict["category"] = category_filter
            if level_filter:
                filter_dict["level"] = level_filter
            search_kwargs["filter"] = filter_dict
        
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    llm = ChatOpenAI(temperature=0)
    
    # Test different filters
    test_cases = [
        {"question": "What should a beginner learn?", "filters": {"level_filter": "beginner"}},
        {"question": "Tell me about programming languages.", "filters": {"category_filter": "programming"}},
        {"question": "What's good for web development?", "filters": {"category_filter": "web"}},
        {"question": "What technologies exist?", "filters": {}}  # No filter
    ]
    
    for test_case in test_cases:
        print(f"\nQuestion: {test_case['question']}")
        if test_case['filters']:
            print(f"Filters: {test_case['filters']}")
        
        # Create retriever with filters
        retriever = create_filtered_retriever(**test_case['filters'])
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": test_case['question']})
        print(f"Answer: {result['result']}")
        
        # Show what documents were used
        if result['source_documents']:
            categories = [doc.metadata['category'] for doc in result['source_documents']]
            print(f"Used documents from categories: {list(set(categories))}")

# =============================================================================
# EXAMPLE 5: RetrievalQA Error Handling
# =============================================================================

def retrievalqa_error_handling():
    """
    What happens when RetrievalQA can't find good answers?
    How to handle errors gracefully.
    """
    print("\n=== EXAMPLE 5: RetrievalQA Error Handling ===")
    
    # Create documents about animals
    animal_docs = [
        Document(page_content="Lions live in Africa and are known as the king of the jungle."),
        Document(page_content="Penguins live in Antarctica and can't fly but are excellent swimmers."),
        Document(page_content="Elephants are the largest land animals and have excellent memories.")
    ]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(animal_docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Create prompt that handles "don't know" cases
    careful_prompt = PromptTemplate(
        template="""
        Use the following context to answer the question. 
        If the context doesn't contain information to answer the question, say "I don't have enough information to answer that question based on the provided context."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": careful_prompt},
        return_source_documents=True
    )
    
    # Ask questions - some answerable, some not
    test_questions = [
        "Where do lions live?",  # Should be answerable
        "What do penguins eat?",  # Not in our documents
        "How big are elephants?",  # Partially answerable
        "What's the weather like in Tokyo?"  # Completely unrelated
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            result = qa_chain.invoke({"query": question})
            print(f"Answer: {result['result']}")
            
            # Show retrieval quality
            if result['source_documents']:
                print(f"Retrieved {len(result['source_documents'])} documents")
            else:
                print("No documents retrieved")
                
        except Exception as e:
            print(f"Error: {e}")

# =============================================================================
# SIMPLE EXPLANATION OF RETRIEVALQA COMPONENTS
# =============================================================================

def explain_retrievalqa_components():
    """
    Let's break down what each part of RetrievalQA does in simple terms.
    """
    print("\n=== RETRIEVALQA COMPONENTS EXPLAINED ===")
    
    print("""
    RetrievalQA has these main parts:
    
    1. 📚 DOCUMENTS: Your knowledge base (like books in a library)
    2. 🔍 EMBEDDINGS: Convert text to numbers for searching
    3. 🗄️ VECTOR STORE: Database that can find similar content quickly
    4. 🎯 RETRIEVER: The "librarian" that finds relevant documents
    5. 🤖 LLM: The "smart assistant" that reads and answers
    6. 🔗 CHAIN: Connects everything together
    
    The flow is:
    Question → Retriever finds docs → LLM reads docs → Generates answer
    """)
    
    # Simple example to show the flow
    print("\n--- Simple Flow Example ---")
    
    # 1. Documents (knowledge base)
    docs = [Document(page_content="Apples are red or green fruits that grow on trees.")]
    print("1. Documents: Created knowledge base about apples")
    
    # 2. Embeddings (convert to searchable format)
    embeddings = OpenAIEmbeddings()
    print("2. Embeddings: Ready to convert text to numbers")
    
    # 3. Vector store (searchable database)
    vectorstore = Chroma.from_documents(docs, embeddings)
    print("3. Vector Store: Created searchable database")
    
    # 4. Retriever (searcher)
    retriever = vectorstore.as_retriever()
    print("4. Retriever: Ready to find relevant documents")
    
    # 5. LLM (answerer)
    llm = ChatOpenAI(temperature=0)
    print("5. LLM: Ready to generate answers")
    
    # 6. Chain (connects everything)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("6. Chain: Connected all components")
    
    # Test the flow
    question = "What color are apples?"
    print(f"\nQuestion: {question}")
    try:
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")
        print("✅ Flow completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")

# =============================================================================
# PRACTICAL TIPS FOR USING RETRIEVALQA
# =============================================================================

def retrievalqa_tips():
    """
    Practical tips for using RetrievalQA effectively.
    """
    print("\n=== PRACTICAL TIPS FOR RETRIEVALQA ===")
    
    tips = """
    💡 TIPS FOR BETTER RETRIEVALQA:
    
    1. 📝 DOCUMENT QUALITY:
       - Keep documents focused on one topic
       - Remove unnecessary formatting
       - Include important keywords
    
    2. 🔢 CHUNK SIZE:
       - Small chunks (200-500 words) for specific questions
       - Larger chunks (500-1000 words) for complex topics
    
    3. 🎯 RETRIEVAL SETTINGS:
       - Start with k=3-5 documents
       - Use similarity search for most cases
       - Try MMR for diverse results
    
    4. 📋 PROMPTS:
       - Be specific about what you want
       - Include instructions for "don't know" cases
       - Ask for sources when needed
    
    5. 🚀 PERFORMANCE:
       - Use smaller, focused document sets
       - Cache embeddings when possible
       - Test different embedding models
    
    6. 🔍 TESTING:
       - Try questions you know the answers to
       - Test edge cases and unclear questions
       - Check if retrieved documents are relevant
    """
    
    print(tips)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎓 RETRIEVALQA EXPLAINED SIMPLY")
    print("=" * 50)
    
    # Run all examples
    examples = [
        ("Basic Example", basic_retrievalqa_example),
        ("Custom Prompt", custom_prompt_retrievalqa),
        ("Chain Types", chain_types_comparison),
        ("With Filtering", retrievalqa_with_filtering),
        ("Error Handling", retrievalqa_error_handling),
        ("Components Explained", explain_retrievalqa_components),
        ("Practical Tips", retrievalqa_tips)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🔸 Running: {name}")
            example_func()
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            print("Make sure you have set your OpenAI API key!")
    
    print("\n🎉 All examples completed!")
    print("\nQuick Summary:")
    print("RetrievalQA = Find relevant documents + Generate answer from those documents")
    print("It's like having a smart assistant that reads your documents and answers questions!")

# =============================================================================
# BONUS: INTERACTIVE RETRIEVALQA
# =============================================================================

def interactive_retrievalqa():
    """
    An interactive example where you can ask questions.
    """
    print("\n=== INTERACTIVE RETRIEVALQA ===")
    
    # Create a simple knowledge base
    knowledge_base = [
        Document(page_content="RetrievalQA is a LangChain component that combines document retrieval with question answering."),
        Document(page_content="LangChain is a framework for building applications with large language models."),
        Document(page_content="Vector stores like Chroma and FAISS help find similar documents quickly."),
        Document(page_content="Embeddings convert text into numerical vectors for similarity comparison."),
        Document(page_content="The 'stuff' chain type puts all retrieved documents into a single prompt.")
    ]
    
    # Set up RetrievalQA
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(knowledge_base, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    print("Knowledge base loaded! You can ask questions about:")
    print("- RetrievalQA")
    print("- LangChain") 
    print("- Vector stores")
    print("- Embeddings")
    print("\nType 'quit' to exit")
    
    while True:
        try:
            question = input("\n🤔 Your question: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            result = qa_chain.invoke({"query": question})
            print(f"🤖 Answer: {result['result']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Uncomment the line below to run the interactive example
# interactive_retrievalqa()