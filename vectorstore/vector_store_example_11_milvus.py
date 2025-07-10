# vector_store_example_11_milvus.py
"""
Basic Milvus + LangChain Example
- Connect to Milvus
- Add documents
- Perform similarity search
- Use a retriever
"""

from langchain.vectorstores import Milvus
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

# 3. Create Milvus vector store (assumes local Milvus at localhost:19530)
vector_store = Milvus.from_documents(
    documents,
    embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="LangChainDemo"
)

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