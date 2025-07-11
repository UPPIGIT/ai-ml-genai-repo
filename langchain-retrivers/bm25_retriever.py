"""
BM25 Retriever
==============

BM25 (Best Matching 25) is a keyword-based retrieval algorithm that's excellent for
exact keyword matching. It's based on TF-IDF but with improvements for document length
normalization and term frequency saturation.

Key Features:
- Excels at exact keyword matching
- Fast and efficient
- No need for embeddings
- Good for FAQ systems, exact term searches
- Complements vector search well in hybrid systems

When to use:
- When users search with specific keywords
- For FAQ or documentation retrieval
- When semantic similarity isn't as important as exact matches
- As part of a hybrid retrieval system
"""

from langchain.retrievers import BM25Retriever
from langchain.docstore.document import Document
from typing import List
import re

class EnhancedBM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever with custom parameters.
        
        Args:
            k1: Controls term frequency saturation (default: 1.2)
            b: Controls document length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.documents = []
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better BM25 performance.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_retriever(self, documents: List[str], preprocess: bool = True):
        """
        Create BM25 retriever from documents.
        
        Args:
            documents: List of document texts
            preprocess: Whether to preprocess documents
        """
        # Preprocess documents if requested
        if preprocess:
            processed_docs = [self.preprocess_text(doc) for doc in documents]
        else:
            processed_docs = documents
            
        # Create Document objects
        self.documents = [
            Document(page_content=doc, metadata={"original": orig})
            for doc, orig in zip(processed_docs, documents)
        ]
        
        # Create BM25 retriever
        self.retriever = BM25Retriever.from_documents(
            self.documents,
            k=3  # Return top 3 results by default
        )
        
        # Set custom parameters if provided
        if hasattr(self.retriever, 'k1'):
            self.retriever.k1 = self.k1
        if hasattr(self.retriever, 'b'):
            self.retriever.b = self.b
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Retriever not created. Call create_retriever() first.")
            
        # Set k if provided
        if k is not None:
            self.retriever.k = k
            
        return self.retriever.get_relevant_documents(query)
    
    def get_scores(self, query: str) -> List[tuple]:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Search query
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        if not self.retriever:
            raise ValueError("Retriever not created.")
            
        # This is a simplified version - actual BM25 scoring would require
        # access to the internal BM25 implementation
        results = self.retrieve(query, k=len(self.documents))
        
        # Return with placeholder scores (in real implementation, 
        # you'd calculate actual BM25 scores)
        return [(doc, 1.0 - i*0.1) for i, doc in enumerate(results)]

class HybridBM25Retriever:
    """
    A hybrid retriever that combines BM25 with additional features.
    """
    def __init__(self):
        self.bm25_retriever = None
        self.documents = []
        
    def add_metadata_boost(self, documents: List[str], metadata_list: List[dict]):
        """
        Create retriever with metadata boosting.
        
        Args:
            documents: List of document texts
            metadata_list: List of metadata dictionaries
        """
        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(documents, metadata_list)
        ]
        
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        
    def retrieve_with_metadata_filter(self, query: str, metadata_filter: dict, k: int = 3):
        """
        Retrieve documents with metadata filtering.
        
        Args:
            query: Search query
            metadata_filter: Dictionary of metadata filters
            k: Number of documents to return
            
        Returns:
            Filtered and ranked documents
        """
        # Get all results
        all_results = self.bm25_retriever.get_relevant_documents(query)
        
        # Filter by metadata
        filtered_results = []
        for doc in all_results:
            match = True
            for key, value in metadata_filter.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_results.append(doc)
                
        return filtered_results[:k]

# Example Usage
if __name__ == "__main__":
    # Sample documents for different use cases
    faq_documents = [
        "How do I reset my password? Go to settings and click 'Reset Password'.",
        "What is the return policy? Items can be returned within 30 days.",
        "How do I contact customer support? Email support@company.com or call 1-800-123-4567.",
        "What payment methods do you accept? We accept credit cards, PayPal, and bank transfers.",
        "How long does shipping take? Standard shipping takes 3-5 business days."
    ]
    
    # Create BM25 retriever
    bm25_retriever = EnhancedBM25Retriever()
    bm25_retriever.create_retriever(faq_documents)
    
    # Test keyword-based retrieval
    print("=== BM25 Keyword-Based Retrieval ===")
    query = "password reset"
    results = bm25_retriever.retrieve(query)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
    
    # Test with different query
    print("\n=== Different Query ===")
    query2 = "return policy days"
    results2 = bm25_retriever.retrieve(query2)
    
    print(f"Query: {query2}")
    for i, doc in enumerate(results2):
        print(f"{i+1}. {doc.page_content}")
    
    # Hybrid example with metadata
    print("\n=== Hybrid BM25 with Metadata ===")
    hybrid_retriever = HybridBM25Retriever()
    
    # Documents with metadata
    docs_with_meta = [
        "Python programming tutorial for beginners",
        "Advanced Python concepts and best practices",
        "JavaScript fundamentals and syntax",
        "React.js component development guide",
        "Machine learning with Python and scikit-learn"
    ]
    
    metadata = [
        {"category": "tutorial", "language": "python", "level": "beginner"},
        {"category": "guide", "language": "python", "level": "advanced"},
        {"category": "tutorial", "language": "javascript", "level": "beginner"},
        {"category": "guide", "language": "javascript", "level": "intermediate"},
        {"category": "tutorial", "language": "python", "level": "advanced"}
    ]
    
    hybrid_retriever.add_metadata_boost(docs_with_meta, metadata)
    
    # Search with metadata filter
    filtered_results = hybrid_retriever.retrieve_with_metadata_filter(
        query="Python tutorial",
        metadata_filter={"language": "python", "level": "beginner"},
        k=2
    )
    
    print("Filtered results (Python + beginner):")
    for doc in filtered_results:
        print(f"- {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
