"""
Ensemble Retriever (Hybrid Search)
==================================

The Ensemble Retriever combines multiple retrieval methods to leverage the strengths
of each approach. This is particularly powerful for combining semantic search (vector-based)
with keyword search (BM25), providing both precise keyword matching and semantic understanding.

Key Benefits:
- Combines strengths of different retrieval methods
- Reduces weaknesses of individual retrievers
- Better coverage of different query types
- Improved robustness and recall
- Handles both exact matches and semantic similarity

Common Combinations:
- Vector Store + BM25 (semantic + keyword)
- Multiple vector stores with different embeddings
- Different chunking strategies
- Domain-specific and general retrievers
"""

from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Dict, Any
import numpy as np

class HybridSearchRetriever:
    def __init__(self, embedding_model=None):
        """
        Initialize hybrid search retriever.
        
        Args:
            embedding_model: Embedding model for vector search
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        
    def create_retrievers(self, documents: List[str], vector_weight: float = 0.5, bm25_weight: float = 0.5):
        """
        Create both vector and BM25 retrievers.
        
        Args:
            documents: List of document texts
            vector_weight: Weight for vector retriever (0.0 to 1.0)
            bm25_weight: Weight for BM25 retriever (0.0 to 1.0)
        """
        # Store documents
        self.documents = documents
        
        # Create Document objects
        docs = [Document(page_content=doc) for doc in documents]
        
        # Create vector store and retriever
        self.vector_store = FAISS.from_documents(docs, self.embedding_model)
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(docs, k=5)
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents using ensemble method.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        if not self.ensemble_retriever:
            raise ValueError("Retrievers not created. Call create_retrievers() first.")
        
        # Set k for both retrievers
        self.ensemble_retriever.retrievers[0].search_kwargs["k"] = k
        self.ensemble_retriever.retrievers[1].k = k
        
        return self.ensemble_retriever.get_relevant_documents(query)
    
    def compare_individual_retrievers(self, query: str) -> Dict[str, List[Document]]:
        """
        Compare results from individual retrievers and ensemble.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with results from each retriever
        """
        results = {}
        
        # Vector retriever results
        vector_results = self.vector_store.similarity_search(query, k=5)
        results["vector"] = vector_results
        
        # BM25 retriever results
        bm25_results = self.bm25_retriever.get_relevant_documents(query)
        results["bm25"] = bm25_results
        
        # Ensemble results
        ensemble_results = self.ensemble_retriever.get_relevant_documents(query)
        results["ensemble"] = ensemble_results
        
        return results

class MultiEmbeddingEnsembleRetriever:
    """
    Ensemble retriever using multiple embedding models.
    """
    
    def __init__(self, embedding_models: List[Any]):
        """
        Initialize with multiple embedding models.
        
        Args:
            embedding_models: List of embedding models
        """
        self.embedding_models = embedding_models
        self.vector_stores = []
        self.ensemble_retriever = None
        
    def create_multi_embedding_retriever(self, documents: List[str], weights: List[float] = None):
        """
        Create ensemble retriever with multiple embedding models.
        
        Args:
            documents: List of document texts
            weights: Weights for each embedding model
        """
        if weights is None:
            weights = [1.0 / len(self.embedding_models)] * len(self.embedding_models)
        
        docs = [Document(page_content=doc) for doc in documents]
        retrievers = []
        
        # Create a vector store for each embedding model
        for embedding_model in self.embedding_models:
            vector_store = FAISS.from_documents(docs, embedding_model)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retrievers.append(retriever)
            self.vector_stores.append(vector_store)
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve using multi-embedding ensemble."""
        if not self.ensemble_retriever:
            raise ValueError("Multi-embedding retriever not created.")
        
        return self.ensemble_retriever.get_relevant_documents(query)

class CustomEnsembleRetriever:
    """
    Custom ensemble retriever with advanced fusion strategies.
    """
    
    def __init__(self):
        self.retrievers = []
        self.weights = []
        self.fusion_method = "weighted_sum"
        
    def add_retriever(self, retriever: Any, weight: float = 1.0):
        """
        Add a retriever to the ensemble.
        
        Args:
            retriever: Retriever instance
            weight: Weight for this retriever
        """
        self.retrievers.append(retriever)
        self.weights.append(weight)
    
    def set_fusion_method(self, method: str):
        """
        Set the fusion method for combining results.
        
        Args:
            method: Fusion method ("weighted_sum", "rrf", "max", "mean")
        """
        self.fusion_method = method
    
    def reciprocal_rank_fusion(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Implement Reciprocal Rank Fusion (RRF) for combining ranked lists.
        
        Args:
            ranked_lists: List of ranked document lists
            k: RRF parameter (default: 60)
            
        Returns:
            Fused ranking of documents
        """
        # Create a dictionary to store document scores
        doc_scores = {}
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_key = doc.page_content  # Use content as key for deduplication
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0}
                
                # RRF score: 1 / (k + rank)
                doc_scores[doc_key]["score"] += 1 / (k + rank + 1)
        
        # Sort by score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
    
    def weighted_sum_fusion(self, ranked_lists: List[List[Document]]) -> List[Document]:
        """
        Implement weighted sum fusion.
        
        Args:
            ranked_lists: List of ranked document lists
            
        Returns:
            Fused ranking of documents
        """
        doc_scores = {}
        
        for i, ranked_list in enumerate(ranked_lists):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            
            for rank, doc in enumerate(ranked_list):
                doc_key = doc.page_content
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0}
                
                # Weighted score based on rank and weight
                score = weight * (len(ranked_list) - rank) / len(ranked_list)
                doc_scores[doc_key]["score"] += score
        
        # Sort by score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
    
    def retrieve_with_fusion(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents using custom fusion method.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            Fused results from all retrievers
        """
        if not self.retrievers:
            raise ValueError("No retrievers added to ensemble.")
        
        # Get results from all retrievers
        all_results = []
        for retriever in self.retrievers:
            try:
                if hasattr(retriever, 'k'):
                    retriever.k = k
                elif hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs["k"] = k
                
                results = retriever.get_relevant_documents(query)
                all_results.append(results)
            except Exception as e:
                print(f"Error with retriever: {e}")
                all_results.append([])
        
        # Apply fusion method
        if self.fusion_method == "rrf":
            fused_results = self.reciprocal_rank_fusion(all_results)
        elif self.fusion_method == "weighted_sum":
            fused_results = self.weighted_sum_fusion(all_results)
        else:
            # Default to simple concatenation and deduplication
            seen = set()
            fused_results = []
            for result_list in all_results:
                for doc in result_list:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        fused_results.append(doc)
        
        return fused_results[:k]

class DomainSpecificEnsembleRetriever:
    """
    Ensemble retriever that combines domain-specific and general retrievers.
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.general_retriever = None
        self.domain_retrievers = {}
        self.ensemble_retriever = None
        
    def create_domain_retrievers(self, general_docs: List[str], domain_data: Dict[str, List[str]]):
        """
        Create general and domain-specific retrievers.
        
        Args:
            general_docs: General documents
            domain_data: Dictionary with domain names as keys and documents as values
        """
        # Create general retriever
        general_doc_objects = [Document(page_content=doc) for doc in general_docs]
        general_store = FAISS.from_documents(general_doc_objects, self.embedding_model)
        self.general_retriever = general_store.as_retriever(search_kwargs={"k": 3})
        
        # Create domain-specific retrievers
        for domain, docs in domain_data.items():
            domain_doc_objects = [Document(page_content=doc, metadata={"domain": domain}) for doc in docs]
            domain_store = FAISS.from_documents(domain_doc_objects, self.embedding_model)
            domain_retriever = domain_store.as_retriever(search_kwargs={"k": 3})
            self.domain_retrievers[domain] = domain_retriever
    
    def retrieve_by_domain(self, query: str, domain: str = None) -> List[Document]:
        """
        Retrieve documents, optionally focusing on a specific domain.
        
        Args:
            query: Search query
            domain: Specific domain to focus on (optional)
            
        Returns:
            Retrieved documents
        """
        if domain and domain in self.domain_retrievers:
            # Use domain-specific retriever with higher weight
            retrievers = [self.general_retriever, self.domain_retrievers[domain]]
            weights = [0.3, 0.7]  # Higher weight for domain-specific
        else:
            # Use all retrievers with equal weights
            retrievers = [self.general_retriever] + list(self.domain_retrievers.values())
            weights = [0.5] + [0.5 / len(self.domain_retrievers)] * len(self.domain_retrievers)
        
        ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
        return ensemble.get_relevant_documents(query)

# Example Usage
if __name__ == "__main__":
    # Sample documents for different domains
    general_docs = [
        "Artificial intelligence is a broad field of computer science.",
        "Machine learning is a subset of artificial intelligence.",
        "Programming languages are used to write software applications.",
        "Data structures help organize and store data efficiently."
    ]
    
    tech_docs = [
        "Python is a popular programming language for data science and machine learning.",
        "React is a JavaScript library for building user interfaces.",
        "Docker containers provide lightweight virtualization for applications.",
        "Git is a version control system for tracking code changes."
    ]
    
    science_docs = [
        "Quantum computing uses quantum mechanical phenomena to process information.",
        "CRISPR gene editing technology allows precise DNA modifications.",
        "Climate change affects global weather patterns and ecosystems.",
        "Renewable energy sources include solar, wind, and hydroelectric power."
    ]
    
    all_docs = general_docs + tech_docs + science_docs
    
    print("=== Hybrid Search Retriever Demo ===\n")
    
    # Create hybrid search retriever
    hybrid_retriever = HybridSearchRetriever()
    hybrid_retriever.create_retrievers(all_docs, vector_weight=0.6, bm25_weight=0.4)
    
    # Test query
    query = "machine learning Python programming"
    print(f"Query: {query}")
    print("-" * 50)
    
    # Compare individual retrievers
    try:
        comparison = hybrid_retriever.compare_individual_retrievers(query)
        
        for method, docs in comparison.items():
            print(f"\n{method.upper()} RESULTS:")
            for i, doc in enumerate(docs[:3], 1):
                print(f"{i}. {doc.page_content}")
            print()
    except Exception as e:
        print(f"Error in hybrid retrieval: {e}")
    
    # Custom ensemble retriever demo
    print("\n=== Custom Ensemble Retriever Demo ===\n")
    
    try:
        custom_ensemble = CustomEnsembleRetriever()
        
        # Create individual retrievers
        docs = [Document(page_content=doc) for doc in all_docs]
        
        # Vector retriever
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(docs, k=5)
        
        # Add retrievers to ensemble
        custom_ensemble.add_retriever(vector_retriever, weight=0.6)
        custom_ensemble.add_retriever(bm25_retriever, weight=0.4)
        
        # Test different fusion methods
        for fusion_method in ["weighted_sum", "rrf"]:
            print(f"{fusion_method.upper()} FUSION:")
            custom_ensemble.set_fusion_method(fusion_method)
            
            results = custom_ensemble.retrieve_with_fusion(query, k=3)
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.page_content}")
            print()
    
    except Exception as e:
        print(f"Error in custom ensemble: {e}")
    
    # Domain-specific ensemble demo
    print("\n=== Domain-Specific Ensemble Demo ===\n")
    
    try:
        domain_ensemble = DomainSpecificEnsembleRetriever()
        domain_data = {
            "technology": tech_docs,
            "science": science_docs
        }
        
        domain_ensemble.create_domain_retrievers(general_docs, domain_data)
        
        # Test general query
        general_query = "What is artificial intelligence?"
        print(f"General Query: {general_query}")
        general_results = domain_ensemble.retrieve_by_domain(general_query)
        
        for i, doc in enumerate(general_results[:3], 1):
            domain = doc.metadata.get("domain", "general")
            print(f"{i}. [{domain}] {doc.page_content}")
        
        # Test domain-specific query
        tech_query = "How to use Python for programming?"
        print(f"\nTech Query: {tech_query}")
        tech_results = domain_ensemble.retrieve_by_domain(tech_query, domain="technology")
        
        for i, doc in enumerate(tech_results[:3], 1):
            domain = doc.metadata.get("domain", "general")
            print(f"{i}. [{domain}] {doc.page_content}")
    
    except Exception as e:
        print(f"Error in domain-specific ensemble