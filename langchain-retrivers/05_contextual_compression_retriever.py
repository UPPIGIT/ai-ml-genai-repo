"""
05_contextual_compression_retriever.py
Example: Using ContextualCompressionRetriever in LangChain
Step-by-step with comments. Uses HuggingFace open-source embeddings and LLM.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from transformers import pipeline

# 2. Prepare some mock documents
docs = [
    Document(page_content="The capital of France is Paris. Paris is a major European city."),
    Document(page_content="Berlin is the capital of Germany. Berlin has a rich history."),
]

# 3. Create a HuggingFace embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# 4. Create a FAISS vector store and add documents
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create a HuggingFace LLM pipeline (e.g., flan-tiny for demo)
generator = pipeline("text-generation", model="google/flan-tiny")
llm = HuggingFacePipeline(pipeline=generator)
compressor = LLMChainExtractor.from_llm(llm)

# 6. Create a ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# 7. Retrieve and compress documents for a query
query = "What is the capital of France?"
results = compression_retriever.get_relevant_documents(query)

# 8. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}") 