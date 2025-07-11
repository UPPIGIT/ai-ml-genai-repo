"""
08_ensemble_retriever.py
Example: Using EnsembleRetriever in LangChain
Step-by-step with comments. Combines BM25 and VectorStore retrievers.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.ensemble import EnsembleRetriever

# 2. Prepare some mock documents
docs = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="The tallest mountain is Mount Everest."),
    Document(page_content="Python is a popular programming language."),
]

# 3. Create a HuggingFace embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# 4. Create a FAISS vector store and add documents
vectorstore = FAISS.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever()

# 5. Create a BM25Retriever
bm25_retriever = BM25Retriever.from_documents(docs)

# 6. Create the EnsembleRetriever (combine both)
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever])

# 7. Define a query
query = "What is the capital of France?"

# 8. Retrieve relevant documents
results = ensemble_retriever.invoke(query)

# 9. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}") 