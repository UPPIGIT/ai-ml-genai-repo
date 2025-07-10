# vector_store_example_9_chromadb.py
"""
Basic ChromaDB + LangChain Example
- Add documents
- Perform similarity search
- Use a retriever
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Example documents
documents = [
    Document(page_content="The cat sits outside."),
    Document(page_content="A dog barks loudly."),
    Document(page_content="Birds are flying in the sky."),
    Document(page_content="The sun is bright today.")
]

# 3. Create Chroma vector store (in-memory, with persistence)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# 4. Similarity search
query = "Animals outside"
results = vector_store.similarity_search(query, k=2)
print(f"Query: {query}\nTop results:")
for doc in results:
    print(f"  - {doc.page_content}")

# 5. Use retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.get_relevant_documents(query)
print("\nRetriever results:")
for doc in retrieved_docs:
    print(f"  - {doc.page_content}") 