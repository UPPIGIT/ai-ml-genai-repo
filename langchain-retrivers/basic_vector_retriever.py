"""
Basic Vector Store Retriever
============================

This is the most fundamental retriever in LangChain that performs similarity search
using vector embeddings. It's the foundation for most RAG (Retrieval-Augmented Generation) systems.

Key Concepts:
- Documents are converted to embeddings using an embedding model
- Query is also converted to embeddings
- Similarity search finds the most relevant documents
- Supports various distance metrics (cosine, euclidean, etc.)
"""

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.retrievers import VectorStoreRetriever
from typing import List

class BasicVectorRetriever:
    def __init__(self, embedding_model=None):
        """
        Initialize the basic vector retriever.
        
        Args:
            embedding_model: The embedding model to use (defaults to OpenAI)
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None
        
    def create_vector_store(self, documents: List[str], chunk_size: int = 1000):
        """
        Create a vector store from documents.
        
        Args:
            documents: List of document texts
            chunk_size: Size of text chunks for splitting
        """
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separator="\n"
        )
        
        # Create Document objects
        docs = [Document(page_content=doc) for doc in documents]
        
        # Split documents
        split_docs = text_splitter.split_documents(docs)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            split_docs, 
            self.embedding_model
        )
        
        # Create retriever
        self.retriever = VectorStoreRetriever(
            vectorstore=self.vector_store,
            search_kwargs={"k": 3}  # Return top 3 results
        )
        
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Vector store not created. Call create_vector_store() first.")
            
        return self.retriever.get_relevant_documents(query)
    
    def similarity_search_with_scores(self, query: str, k: int = 3):
        """
        Perform similarity search with confidence scores.
        
        Args:
            query: The search query
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not created.")
            
        return self.vector_store.similarity_search_with_score(query, k=k)

# Example Usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "Vector stores are used to store and retrieve documents based on semantic similarity.",
        "Retrievers are components that take a query and return relevant documents.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "OpenAI embeddings convert text into high-dimensional vectors for semantic search."
    ]
    
    # Create retriever
    retriever = BasicVectorRetriever()
    
    # Create vector store
    retriever.create_vector_store(documents)
    
    # Perform retrieval
    query = "What is LangChain?"
    results = retriever.retrieve(query)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
        
    # Get results with scores
    print("\nWith similarity scores:")
    scored_results = retriever.similarity_search_with_scores(query)
    for doc, score in scored_results:
        print(f"Score: {score:.4f} - {doc.page_content}")
