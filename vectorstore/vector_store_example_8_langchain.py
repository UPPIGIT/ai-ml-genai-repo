# vector_store_example_8_langchain.py
"""
Step 8: Integration with LangChain
- Uses LangChain's FAISS vector store
- Uses HuggingFaceEmbeddings for open-source embedding
- Shows how to add documents, search, and use a retriever
"""

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. Set up HuggingFace embeddings (open source)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Example documents
documents = [
    Document(page_content="The cat sits outside."),
    Document(page_content="A dog barks loudly."),
    Document(page_content="Birds are flying in the sky."),
    Document(page_content="The sun is bright today.")
]

# 3. Create FAISS vector store from documents
vector_store = FAISS.from_documents(documents, embeddings)

# 4. Similarity search
query = "Animals outside"
results = vector_store.similarity_search(query, k=2)
print(f"Query: {query}\nTop results:")
for doc in results:
    print(f"  - {doc.page_content}")

# 5. Use retriever for more advanced retrieval
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.get_relevant_documents(query)
print("\nRetriever results:")
for doc in retrieved_docs:
    print(f"  - {doc.page_content}") 