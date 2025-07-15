from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
# Step 2: Initialize the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Step 3: Create a Chroma vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="sample_chroma_db"  # Directory to store the vector store
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# Example query to retrieve information from the vector store
query = "What is LangChain?"
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters of content
    print("\n")


