# LangChain Vector Store Examples: Simple to Advanced

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Simple Example: Basic Text Search](#simple-example-basic-text-search)
3. [Intermediate Example: Document Q&A](#intermediate-example-document-qa)
4. [Advanced Example: Multi-Document RAG with Memory](#advanced-example-multi-document-rag-with-memory)
5. [Production Example: Persistent Vector Store with Custom Embeddings](#production-example-persistent-vector-store-with-custom-embeddings)

## Prerequisites

First, install the required packages:

```bash
pip install langchain langchain-community langchain-huggingface
pip install chromadb faiss-cpu sentence-transformers
pip install transformers torch
```

## Simple Example: Basic Text Search

This example shows how to create a basic vector store and perform similarity search.

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Step 1: Initialize embeddings model (open-source)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Create sample documents
documents = [
    Document(page_content="The cat sat on the mat.", metadata={"source": "story1"}),
    Document(page_content="Dogs are loyal animals.", metadata={"source": "story2"}),
    Document(page_content="Python is a programming language.", metadata={"source": "tech1"}),
    Document(page_content="Machine learning is part of AI.", metadata={"source": "tech2"}),
]

# Step 3: Create vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Step 4: Perform similarity search
query = "What programming language is mentioned?"
results = vector_store.similarity_search(query, k=2)

print("Query:", query)
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
    print(f"Source: {doc.metadata['source']}")
    print("---")
```

### Explanation:
- **Embeddings**: Uses HuggingFace's sentence transformers for free, open-source embeddings
- **Vector Store**: FAISS provides fast similarity search
- **Documents**: Simple text chunks with metadata
- **Search**: Finds most similar documents to the query

## Intermediate Example: Document Q&A

This example shows how to build a question-answering system using documents and an open-source LLM.

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os

# Step 1: Prepare documents
sample_text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create 
intelligent machines. Machine learning is a subset of AI that enables computers 
to learn without being explicitly programmed. Deep learning is a subset of machine 
learning that uses neural networks with multiple layers. Natural Language Processing 
(NLP) is another branch of AI that deals with human language understanding.

Python is widely used in AI development due to its simplicity and extensive libraries 
like TensorFlow, PyTorch, and scikit-learn. These libraries provide tools for building 
and training machine learning models.
"""

# Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

texts = text_splitter.split_text(sample_text)
documents = [Document(page_content=text) for text in texts]

# Step 3: Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Step 4: Initialize open-source LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation",
    model_kwargs={"temperature": 0.1, "max_length": 200}
)

# Step 5: Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# Step 6: Ask questions
questions = [
    "What is machine learning?",
    "Which programming language is popular for AI?",
    "What are some Python libraries for AI?"
]

for question in questions:
    print(f"\nQuestion: {question}")
    result = qa_chain({"query": question})
    print(f"Answer: {result['result']}")
    print("Sources:")
    for doc in result['source_documents']:
        print(f"- {doc.page_content[:100]}...")
    print("="*50)
```

### Key Features:
- **Text Splitting**: Breaks long documents into manageable chunks
- **Persistent Storage**: Chroma saves vectors to disk
- **Open-source LLM**: Uses HuggingFace models for generation
- **Retrieval Chain**: Combines search and generation for Q&A

## Advanced Example: Multi-Document RAG with Memory

This example demonstrates a sophisticated RAG system with conversation memory and multiple document types.

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import tempfile
import os

class AdvancedRAGSystem:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize LLM
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="microsoft/DialoGPT-small",
            task="text-generation",
            model_kwargs={"temperature": 0.2, "max_length": 300}
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        self.vector_store = None
        self.qa_chain = None
    
    def add_documents(self, documents):
        """Add documents to the vector store"""
        # Split documents
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ))
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        else:
            new_vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            self.vector_store.merge_from(new_vector_store)
        
        # Create conversational chain
        self._create_chain()
    
    def _create_chain(self):
        """Create the conversational retrieval chain"""
        # Custom prompt template
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain with memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def query(self, question):
        """Query the system with conversation memory"""
        if self.qa_chain is None:
            return "Please add documents first using add_documents()"
        
        result = self.qa_chain({"question": question})
        return {
            "answer": result["answer"],
            "sources": result["source_documents"]
        }
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.memory.chat_memory.messages

# Usage example
def main():
    # Create the RAG system
    rag_system = AdvancedRAGSystem()
    
    # Sample documents from different domains
    documents = [
        Document(
            page_content="""
            Climate change refers to long-term shifts in global temperatures and weather patterns. 
            While climate change is a natural phenomenon, scientific evidence shows that human activities 
            have been the main driver of climate change since the mid-20th century. The primary cause 
            is the emission of greenhouse gases, particularly carbon dioxide from burning fossil fuels.
            """,
            metadata={"source": "climate_science.txt", "topic": "environment"}
        ),
        Document(
            page_content="""
            Renewable energy sources include solar, wind, hydroelectric, and geothermal power. 
            These energy sources are sustainable because they are naturally replenished and don't 
            produce greenhouse gas emissions during operation. Solar panels convert sunlight into 
            electricity, while wind turbines harness wind energy. Hydroelectric plants use flowing 
            water to generate power.
            """,
            metadata={"source": "renewable_energy.txt", "topic": "energy"}
        ),
        Document(
            page_content="""
            Artificial intelligence and machine learning are transforming various industries. 
            AI applications include natural language processing, computer vision, and robotics. 
            Machine learning algorithms can learn from data to make predictions or decisions. 
            Deep learning, a subset of machine learning, uses neural networks to process complex data.
            """,
            metadata={"source": "ai_overview.txt", "topic": "technology"}
        )
    ]
    
    # Add documents to the system
    rag_system.add_documents(documents)
    
    # Interactive conversation
    questions = [
        "What is climate change?",
        "How do renewable energy sources help with climate change?",
        "Can AI help in addressing environmental challenges?",
        "What did we discuss about renewable energy earlier?"
    ]
    
    print("=== Advanced RAG System Demo ===\n")
    
    for question in questions:
        print(f"Question: {question}")
        result = rag_system.query(question)
        print(f"Answer: {result['answer']}")
        print("Sources:")
        for i, doc in enumerate(result['sources']):
            print(f"  {i+1}. {doc.metadata['source']} ({doc.metadata['topic']})")
            print(f"     {doc.page_content[:100]}...")
        print("="*60)

if __name__ == "__main__":
    main()
```

### Advanced Features:
- **Conversation Memory**: Maintains context across multiple questions
- **Multi-Document Support**: Handles documents from different domains
- **Custom Prompts**: Tailored prompts for better responses
- **Metadata Handling**: Tracks document sources and topics
- **Modular Design**: Clean, reusable class structure

## Production Example: Persistent Vector Store with Custom Embeddings

This example shows a production-ready setup with persistent storage and custom embedding strategies.

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import chromadb
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddings(Embeddings):
    """Custom embeddings class with caching and batch processing"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        self.batch_size = 32
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batch processing"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Check cache first
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                if text in self.cache:
                    batch_embeddings.append(self.cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Process uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts)
                for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                    self.cache[text] = embedding.tolist()
                    batch_embeddings.insert(idx, embedding.tolist())
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode([text])[0]
        self.cache[text] = embedding.tolist()
        return embedding.tolist()

class ProductionVectorStore:
    """Production-ready vector store with persistence and monitoring"""
    
    def __init__(self, persist_directory: str = "./production_db"):
        self.persist_directory = persist_directory
        self.embeddings = CustomEmbeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Load existing vector store if available
        self._load_existing_store()
    
    def _load_existing_store(self):
        """Load existing vector store from disk"""
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Could not load existing store: {e}")
            self.vector_store = None
    
    def add_documents_from_directory(self, directory_path: str, glob_pattern: str = "*.txt"):
        """Add documents from a directory"""
        try:
            # Load documents
            loader = DirectoryLoader(directory_path, glob=glob_pattern)
            documents = loader.load()
            
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            
            # Process documents
            self._process_and_store_documents(documents)
            
        except Exception as e:
            logger.error(f"Error loading documents from directory: {e}")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        self._process_and_store_documents(documents)
    
    def _process_and_store_documents(self, documents: List[Document]):
        """Process and store documents in the vector store"""
        # Split documents
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                        "chunk_size": len(chunk)
                    }
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Add to existing store
            self.vector_store.add_documents(all_chunks)
        
        # Persist to disk
        self.vector_store.persist()
        logger.info("Vector store updated and persisted")
    
    def search(self, query: str, k: int = 5, filter_dict: dict = None):
        """Search the vector store with optional filtering"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")
        
        # Perform search
        if filter_dict:
            results = self.vector_store.similarity_search(
                query, k=k, filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def search_with_scores(self, query: str, k: int = 5):
        """Search with similarity scores"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def get_stats(self):
        """Get statistics about the vector store"""
        if self.vector_store is None:
            return {"status": "empty", "document_count": 0}
        
        collection = self.vector_store._collection
        count = collection.count()
        
        return {
            "status": "ready",
            "document_count": count,
            "embedding_cache_size": len(self.embeddings.cache),
            "persist_directory": self.persist_directory
        }
    
    def delete_documents(self, filter_dict: dict):
        """Delete documents based on metadata filter"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        # Note: This requires chromadb with delete functionality
        try:
            self.vector_store.delete(filter=filter_dict)
            self.vector_store.persist()
            logger.info(f"Deleted documents matching filter: {filter_dict}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

# Usage example
def main():
    # Initialize production vector store
    vector_store = ProductionVectorStore("./production_vectordb")
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="""
            Large Language Models (LLMs) are neural networks trained on vast amounts of text data. 
            They can generate human-like text, answer questions, and perform various language tasks. 
            Examples include GPT-3, BERT, and T5. These models use transformer architecture and 
            attention mechanisms to understand context and generate coherent responses.
            """,
            metadata={"source": "llm_basics.txt", "category": "AI", "date": "2024-01-15"}
        ),
        Document(
            page_content="""
            Vector databases are specialized databases designed to store and query high-dimensional 
            vectors efficiently. They use techniques like approximate nearest neighbor search and 
            indexing to enable fast similarity searches. Popular vector databases include Pinecone, 
            Weaviate, and Chroma. They're essential for applications like recommendation systems 
            and retrieval-augmented generation.
            """,
            metadata={"source": "vector_db.txt", "category": "Database", "date": "2024-01-20"}
        ),
        Document(
            page_content="""
            Retrieval-Augmented Generation (RAG) combines the power of pre-trained language models 
            with external knowledge retrieval. The process involves retrieving relevant documents 
            from a knowledge base and using them to generate more accurate and contextual responses. 
            RAG helps reduce hallucinations and enables models to access up-to-date information.
            """,
            metadata={"source": "rag_explained.txt", "category": "AI", "date": "2024-01-25"}
        )
    ]
    
    # Add documents
    vector_store.add_documents(sample_docs)
    
    # Get statistics
    stats = vector_store.get_stats()
    print(f"Vector Store Stats: {stats}")
    
    # Perform searches
    print("\n=== Basic Search ===")
    results = vector_store.search("What are Large Language Models?", k=2)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("---")
    
    print("\n=== Search with Scores ===")
    results_with_scores = vector_store.search_with_scores("vector database", k=2)
    for (doc, score) in results_with_scores:
        print(f"Score: {score:.3f}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata['source']}")
        print("---")
    
    print("\n=== Filtered Search ===")
    ai_results = vector_store.search(
        "machine learning models", 
        k=3, 
        filter_dict={"category": "AI"}
    )
    for doc in ai_results:
        print(f"AI Document: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:150]}...")
        print("---")

if __name__ == "__main__":
    main()
```

### Production Features:
- **Persistent Storage**: Automatically saves and loads vector stores
- **Custom Embeddings**: Optimized embeddings with caching and batch processing
- **Monitoring**: Comprehensive logging and statistics
- **Filtering**: Advanced search with metadata filtering
- **Batch Processing**: Efficient handling of large document sets
- **Error Handling**: Robust error handling and recovery

## Key Takeaways

1. **Start Simple**: Begin with basic FAISS vector stores for prototyping
2. **Choose the Right Storage**: Use Chroma for persistence, FAISS for speed
3. **Optimize Embeddings**: Consider custom embedding strategies for production
4. **Add Memory**: Implement conversation memory for better user experience
5. **Monitor Performance**: Track vector store statistics and search performance
6. **Handle Metadata**: Use metadata for filtering and source tracking
7. **Scale Gradually**: Start with small datasets and scale up systematically

Each example builds upon the previous one, demonstrating increasingly sophisticated techniques for building production-ready RAG systems with LangChain and open-source models.