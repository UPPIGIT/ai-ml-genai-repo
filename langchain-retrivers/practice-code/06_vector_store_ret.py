"""
01_vectorstore_retriever.py
Basic example: Using a Vector Store Retriever (Chroma) in LangChain
Step-by-step with comments. Uses HuggingFace open-source embeddings.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 2. Prepare some mock documents
docs = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="The tallest mountain is Mount Everest."),
    Document(page_content="Python is a popular programming language."),
]

# 3. Create a HuggingFace embedding function (open-source model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# 4. Create a Chroma vector store and add documents
vectorstore = Chroma.from_documents(docs, embeddings)

# 5. Create a VectorStoreRetriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve top 2 results

# 6. Retrieve relevant documents for a query
query = "What is the capital of France?"
results = retriever.invoke(query)

# 7. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}") 