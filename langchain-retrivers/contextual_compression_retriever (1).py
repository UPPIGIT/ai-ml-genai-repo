"""
Contextual Compression Retriever
================================

The Contextual Compression Retriever solves the problem of retrieving relevant documents
that may contain irrelevant information. It retrieves documents first, then compresses
them to extract only the most relevant parts for the given query.

Key Benefits:
- Reduces noise in retrieved documents
- Improves relevance of returned content
- Saves tokens in downstream LLM processing
- Provides more focused context
- Better performance in QA tasks

How it works:
1. Uses a base retriever to get initial documents
2. Passes each document through a compressor
3. Compressor extracts relevant sentences/paragraphs
4. Returns compressed, relevant content only

Types of Compressors:
- LLMChainExtractor: Uses LLM to extract relevant parts
- LLMChainFilter: Uses LLM to filter documents
- EmbeddingsFilter: Uses embeddings to filter content
- DocumentCompressor: Custom compression logic
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any

class ContextualCompressionDemo:
    def __init__(self, llm=None, embedding_model=None):
        """
        Initialize the contextual compression retriever demo.
        
        Args:
            llm: Language model for compression
            embedding_model: Embedding model for vector search
        """
        self.llm = llm or OpenAI(temperature=0)
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store = None
        self.base_retriever = None
        
    def create_vector_store(self, documents: List[str]):
        """
        Create vector store from documents.
        
        Args:
            documents: List of document texts
        """
        # Create Document objects
        docs = [Document(page_content=doc) for doc in documents]
        
        # Create vector store
        self.vector_store = FAISS.from_documents(docs, self.embedding_model)
        
        # Create base retriever
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Get more docs for compression
        )
    
    def create_llm_extractor_retriever(self) -> ContextualCompressionRetriever:
        """
        Create a retriever that uses LLM to extract relevant parts.
        
        Returns:
            ContextualCompressionRetriever with LLM extractor
        """
        compressor = LLMChainExtractor.from_llm(self.llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
    
    def create_llm_filter_retriever(self) -> ContextualCompressionRetriever:
        """
        Create a retriever that uses LLM to filter documents.
        
        Returns:
            ContextualCompressionRetriever with LLM filter
        """
        compressor = LLMChainFilter.from_llm(self.llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
    
    def create_embeddings_filter_retriever(self, similarity_threshold: float = 0.8) -> ContextualCompressionRetriever:
        """
        Create a retriever that uses embeddings to filter content.
        
        Args:
            similarity_threshold: Minimum similarity score for filtering
            
        Returns:
            ContextualCompressionRetriever with embeddings filter
        """
        compressor = EmbeddingsFilter(
            embeddings=self.embedding_model,
            similarity_threshold=similarity_threshold
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
    
    def compare_retrievers(self, query: str) -> Dict[str, List[Document]]:
        """
        Compare different retrieval methods.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with results from different retrievers
        """
        results = {}
        
        # Base retriever (no