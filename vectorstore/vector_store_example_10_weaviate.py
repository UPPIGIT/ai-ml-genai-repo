# vector_store_example_10_weaviate.py
"""
Basic Weaviate + LangChain Example
- Connect to Weaviate
- Add documents
- Perform similarity search
- Use a retriever
"""

import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 1. Connect to Weaviate (assumes local instance at http://localhost:8080)
client = weaviate.Client("http://localhost:8080")

# 2. Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Example documents
documents = [
    Document(page_content="The cat sits outside."),
    Document(page_content="A dog barks loudly."),
    Document(page_content="Birds are flying in the sky."),
    Document(page_content="The sun is bright today.")
]

# 4. Create Weaviate vector store
vector_store = Weaviate.from_documents(
    documents,
    embeddings,
    client=client,
    index_name="LangChainDemo",
    by_text=False  # Set to True if you want Weaviate to generate embeddings
)

# 5. Similarity search
query = "Animals outside"
results = vector_store.similarity_search(query, k=2)
print(f"Query: {query}\nTop results:")
for doc in results:
    print(f"  - {doc.page_content}")

# 6. Use retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.get_relevant_documents(query)
print("\nRetriever results:")
for doc in retrieved_docs:
    print(f"  - {doc.page_content}") 