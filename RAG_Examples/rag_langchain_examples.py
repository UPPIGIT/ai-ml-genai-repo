# 5 RAG Examples with LangChain - Step by Step Implementation

"""
This document contains 5 comprehensive examples of Retrieval-Augmented Generation (RAG) 
using LangChain, demonstrating different approaches and use cases.

Requirements:
pip install langchain langchain-openai langchain-community chromadb faiss-cpu pypdf2 sentence-transformers
"""

import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# =============================================================================
# EXAMPLE 1: Basic RAG with Text Documents
# =============================================================================

def example_1_basic_text_rag():
    """
    Basic RAG implementation with text documents using Chroma vector store.
    This example demonstrates the fundamental RAG pipeline.
    """
    print("=== Example 1: Basic RAG with Text Documents ===")
    
    # Step 1: Prepare sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity and readability.", 
                metadata={"source": "python_intro.txt"}),
        Document(page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.", 
                metadata={"source": "ml_basics.txt"}),
        Document(page_content="RAG (Retrieval-Augmented Generation) combines retrieval and generation for better AI responses.", 
                metadata={"source": "rag_explanation.txt"}),
        Document(page_content="LangChain is a framework for developing applications powered by language models.", 
                metadata={"source": "langchain_info.txt"}),
        Document(page_content="Vector databases store high-dimensional vectors for efficient similarity search.", 
                metadata={"source": "vector_db.txt"})
    ]
    
    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    # Step 3: Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="basic_rag_collection"
    )
    
    # Step 4: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Retrieve top 2 most similar chunks
    )
    
    # Step 5: Create custom prompt template
    prompt_template = """
    Use the following context to answer the question. If you don't know the answer based on the context, say so.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Step 6: Create RAG chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    # Step 7: Test the RAG system
    queries = [
        "What is Python?",
        "How does RAG work?",
        "What is machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag_chain.invoke({"query": query})
        print(f"Answer: {result['result']}")
        print(f"Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")

# =============================================================================
# EXAMPLE 2: RAG with PDF Documents
# =============================================================================

def example_2_pdf_rag():
    """
    RAG implementation with PDF documents using FAISS vector store.
    Demonstrates document loading and processing from PDF files.
    """
    print("\n=== Example 2: RAG with PDF Documents ===")
    
    # Step 1: Load PDF documents
    # Note: Replace with actual PDF file paths
    pdf_paths = ["document1.pdf", "document2.pdf"]  # Add your PDF paths here
    
    documents = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        else:
            print(f"Warning: {pdf_path} not found, creating sample document")
            # Create sample document for demonstration
            documents.append(Document(
                page_content=f"This is sample content from {pdf_path}. It contains information about various topics.",
                metadata={"source": pdf_path, "page": 1}
            ))
    
    # Step 2: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # Step 3: Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Step 4: Save vector store for later use
    vectorstore.save_local("faiss_pdf_index")
    
    # Step 5: Create retriever with different search parameters
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    
    # Step 6: Create enhanced prompt template
    prompt_template = """
    You are an assistant that answers questions based on provided documents.
    Use the following context to provide a comprehensive answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Provide a detailed answer based on the context
    - If information is not available in the context, clearly state that
    - Include relevant details from the source documents
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Step 7: Create RAG chain
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    # Step 8: Test with sample queries
    sample_queries = [
        "What are the main topics covered in these documents?",
        "Can you provide a summary of the key points?"
    ]
    
    for query in sample_queries:
        print(f"\nQuery: {query}")
        result = rag_chain.invoke({"query": query})
        print(f"Answer: {result['result']}")
        print(f"Number of source documents: {len(result['source_documents'])}")

# =============================================================================
# EXAMPLE 3: Conversational RAG with Memory
# =============================================================================

def example_3_conversational_rag():
    """
    Conversational RAG with memory to maintain context across interactions.
    Demonstrates multi-turn conversations with document retrieval.
    """
    print("\n=== Example 3: Conversational RAG with Memory ===")
    
    # Step 1: Create knowledge base
    knowledge_base = [
        Document(page_content="Climate change refers to long-term shifts in temperatures and weather patterns.", 
                metadata={"topic": "climate", "source": "climate_basics.txt"}),
        Document(page_content="Renewable energy sources include solar, wind, hydro, and geothermal power.", 
                metadata={"topic": "energy", "source": "renewable_energy.txt"}),
        Document(page_content="Carbon footprint is the total amount of greenhouse gases produced by human activities.", 
                metadata={"topic": "environment", "source": "carbon_footprint.txt"}),
        Document(page_content="Sustainable development meets present needs without compromising future generations.", 
                metadata={"topic": "sustainability", "source": "sustainability.txt"}),
        Document(page_content="Electric vehicles produce zero direct emissions and are becoming more popular.", 
                metadata={"topic": "transportation", "source": "electric_vehicles.txt"})
    ]
    
    # Step 2: Split and create vector store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(knowledge_base)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="conversational_rag"
    )
    
    # Step 3: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    # Step 4: Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Step 5: Create conversational RAG chain
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
    conversational_rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    # Step 6: Simulate conversation
    conversation_queries = [
        "What is climate change?",
        "How can renewable energy help with this issue?",
        "What about electric vehicles? Are they related to what we discussed?",
        "Can you summarize our conversation so far?"
    ]
    
    print("\nStarting conversational RAG session:")
    for i, query in enumerate(conversation_queries, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {query}")
        
        result = conversational_rag_chain.invoke({"question": query})
        print(f"Assistant: {result['answer']}")
        
        # Show source documents for transparency
        if result.get('source_documents'):
            sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
            print(f"Sources: {list(set(sources))}")

# =============================================================================
# EXAMPLE 4: RAG with Web Content
# =============================================================================

def example_4_web_rag():
    """
    RAG implementation that retrieves information from web sources.
    Demonstrates loading content from URLs and processing web data.
    """
    print("\n=== Example 4: RAG with Web Content ===")
    
    # Step 1: Load web content
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
    ]
    
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["url"] = url
                documents.append(doc)
        except Exception as e:
            print(f"Could not load {url}: {e}")
            # Create sample document for demonstration
            documents.append(Document(
                page_content=f"This is sample content about AI and ML from {url}. It covers various aspects of artificial intelligence and machine learning technologies.",
                metadata={"url": url, "source": "web"}
            ))
    
    # Step 2: Split documents with web-specific settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # Step 3: Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="web_rag"
    )
    
    # Step 4: Create retriever with threshold
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 3}
    )
    
    # Step 5: Create specialized prompt for web content
    web_prompt_template = """
    You are an AI assistant that provides information based on web sources.
    Use the following context from web pages to answer the question accurately.
    
    Context from web sources:
    {context}
    
    Question: {question}
    
    Instructions:
    - Provide a comprehensive answer based on the web content
    - Mention if information comes from specific sources when relevant
    - Be factual and precise
    - If the context doesn't contain enough information, state that clearly
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=web_prompt_template,
        input_variables=["context", "question"]
    )
    
    # Step 6: Create RAG chain
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    web_rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    # Step 7: Test with web-specific queries
    web_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of natural language processing?"
    ]
    
    for query in web_queries:
        print(f"\nQuery: {query}")
        result = web_rag_chain.invoke({"query": query})
        print(f"Answer: {result['result']}")
        
        # Show source URLs
        if result.get('source_documents'):
            urls = [doc.metadata.get('url', 'Unknown') for doc in result['source_documents']]
            print(f"Sources: {list(set(urls))}")

# =============================================================================
# EXAMPLE 5: Advanced RAG with Custom Retrieval and Reranking
# =============================================================================

def example_5_advanced_rag():
    """
    Advanced RAG with custom retrieval strategies and reranking.
    Demonstrates sophisticated retrieval techniques and result processing.
    """
    print("\n=== Example 5: Advanced RAG with Custom Retrieval ===")
    
    # Step 1: Create diverse knowledge base
    tech_documents = [
        Document(page_content="Kubernetes is an open-source container orchestration platform for automating deployment, scaling, and management of containerized applications.", 
                metadata={"category": "devops", "difficulty": "intermediate", "topic": "kubernetes"}),
        Document(page_content="Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers.", 
                metadata={"category": "devops", "difficulty": "beginner", "topic": "docker"}),
        Document(page_content="Microservices architecture breaks down applications into small, independent services that communicate over well-defined APIs.", 
                metadata={"category": "architecture", "difficulty": "intermediate", "topic": "microservices"}),
        Document(page_content="GraphQL is a query language for APIs that allows clients to request exactly the data they need.", 
                metadata={"category": "api", "difficulty": "intermediate", "topic": "graphql"}),
        Document(page_content="Serverless computing allows developers to build applications without managing server infrastructure.", 
                metadata={"category": "cloud", "difficulty": "beginner", "topic": "serverless"}),
        Document(page_content="Apache Kafka is a distributed streaming platform used for building real-time data pipelines and streaming applications.", 
                metadata={"category": "data", "difficulty": "advanced", "topic": "kafka"}),
        Document(page_content="Redis is an in-memory data structure store used as a database, cache, and message broker.", 
                metadata={"category": "database", "difficulty": "intermediate", "topic": "redis"})
    ]
    
    # Step 2: Advanced text splitting with metadata preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        add_start_index=True
    )
    splits = text_splitter.split_documents(tech_documents)
    
    # Step 3: Create vector store with metadata filtering
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="advanced_rag"
    )
    
    # Step 4: Custom retrieval function with metadata filtering
    def custom_retrieval(query: str, difficulty_filter: str = None, category_filter: str = None) -> List[Document]:
        """Custom retrieval with metadata filtering"""
        search_kwargs = {"k": 5}
        
        # Add metadata filters
        if difficulty_filter or category_filter:
            filter_dict = {}
            if difficulty_filter:
                filter_dict["difficulty"] = difficulty_filter
            if category_filter:
                filter_dict["category"] = category_filter
            search_kwargs["filter"] = filter_dict
        
        # Retrieve documents
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        docs = retriever.get_relevant_documents(query)
        return docs
    
    # Step 5: Custom reranking function
    def rerank_documents(query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """Simple reranking based on query relevance and metadata"""
        scored_docs = []
        
        for doc in documents:
            score = 0
            
            # Basic relevance scoring
            query_words = query.lower().split()
            content_words = doc.page_content.lower().split()
            
            # Count query word matches
            matches = sum(1 for word in query_words if word in content_words)
            score += matches * 2
            
            # Boost score based on difficulty preference
            if "beginner" in query.lower() and doc.metadata.get("difficulty") == "beginner":
                score += 1
            elif "advanced" in query.lower() and doc.metadata.get("difficulty") == "advanced":
                score += 1
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    # Step 6: Advanced RAG chain with custom processing
    class AdvancedRAGChain:
        def __init__(self, llm, vectorstore):
            self.llm = llm
            self.vectorstore = vectorstore
            
        def invoke(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
            # Custom retrieval
            docs = custom_retrieval(
                query, 
                difficulty_filter=filters.get("difficulty") if filters else None,
                category_filter=filters.get("category") if filters else None
            )
            
            # Rerank documents
            reranked_docs = rerank_documents(query, docs, top_k=3)
            
            # Create context
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            # Advanced prompt with metadata
            prompt = f"""
            You are a technical expert assistant. Answer the following question based on the provided context.
            
            Context:
            {context}
            
            Question: {query}
            
            Instructions:
            - Provide a comprehensive, technical answer
            - Include relevant details and examples
            - If the context covers multiple difficulty levels, explain accordingly
            - Mention specific technologies when relevant
            
            Answer:"""
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return {
                "query": query,
                "answer": response.content,
                "source_documents": reranked_docs,
                "metadata": [doc.metadata for doc in reranked_docs]
            }
    
    # Step 7: Test advanced RAG
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
    advanced_rag = AdvancedRAGChain(llm, vectorstore)
    
    test_cases = [
        {
            "query": "What are containerization technologies for beginners?",
            "filters": {"difficulty": "beginner"}
        },
        {
            "query": "How do microservices work?",
            "filters": {"category": "architecture"}
        },
        {
            "query": "What are advanced data streaming technologies?",
            "filters": {"difficulty": "advanced"}
        }
    ]
    
    for test_case in test_cases:
        print(f"\nQuery: {test_case['query']}")
        print(f"Filters: {test_case['filters']}")
        
        result = advanced_rag.invoke(test_case["query"], test_case["filters"])
        print(f"Answer: {result['answer']}")
        print(f"Topics covered: {[meta['topic'] for meta in result['metadata']]}")
        print(f"Difficulty levels: {[meta['difficulty'] for meta in result['metadata']]}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run all RAG examples. Comment out examples you don't want to run.
    Make sure to set your OpenAI API key in the environment variable.
    """
    
    print("Running RAG Examples with LangChain")
    print("=" * 50)
    
    # Example 1: Basic RAG
    try:
        example_1_basic_text_rag()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    # Example 2: PDF RAG
    try:
        example_2_pdf_rag()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    # Example 3: Conversational RAG
    try:
        example_3_conversational_rag()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    # Example 4: Web RAG
    try:
        example_4_web_rag()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    # Example 5: Advanced RAG
    try:
        example_5_advanced_rag()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    print("\nAll examples completed!")

# =============================================================================
# ADDITIONAL UTILITIES AND BEST PRACTICES
# =============================================================================

class RAGEvaluator:
    """Utility class for evaluating RAG performance"""
    
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
    
    def evaluate_relevance(self, query: str, retrieved_docs: List[Document]) -> float:
        """Simple relevance scoring"""
        if not retrieved_docs:
            return 0.0
        
        query_words = set(query.lower().split())
        total_score = 0
        
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / len(query_words) if query_words else 0
            total_score += score
        
        return total_score / len(retrieved_docs)
    
    def evaluate_answer_quality(self, query: str, answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evaluate answer quality based on expected keywords"""
        answer_lower = answer.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        
        return {
            "keyword_coverage": len(found_keywords) / len(expected_keywords),
            "found_keywords": found_keywords,
            "answer_length": len(answer),
            "completeness_score": min(len(answer) / 200, 1.0)  # Normalize to 0-1
        }

# Best Practices Comments:

"""
RAG BEST PRACTICES:

1. Document Preparation:
   - Clean and preprocess documents
   - Remove unnecessary formatting
   - Ensure consistent structure
   - Add meaningful metadata

2. Chunking Strategy:
   - Choose appropriate chunk size (200-1000 tokens)
   - Use semantic chunking when possible
   - Maintain context with overlapping chunks
   - Preserve document structure

3. Embedding Selection:
   - Use domain-specific embeddings when available
   - Consider multilingual embeddings for international content
   - Test different embedding models for your use case

4. Retrieval Optimization:
   - Experiment with different retrieval strategies (similarity, MMR, etc.)
   - Use metadata filtering to improve relevance
   - Implement reranking for better results
   - Consider hybrid retrieval (dense + sparse)

5. Prompt Engineering:
   - Be specific about expected output format
   - Include context instructions
   - Handle cases where information is not available
   - Use system prompts for consistent behavior

6. Evaluation:
   - Implement relevance scoring
   - Test with diverse queries
   - Monitor hallucination rates
   - Evaluate answer completeness

7. Production Considerations:
   - Cache frequently accessed embeddings
   - Implement proper error handling
   - Monitor system performance
   - Plan for scaling vector storage

8. Security:
   - Validate user inputs
   - Sanitize retrieved content
   - Implement access controls
   - Audit sensitive operations
"""