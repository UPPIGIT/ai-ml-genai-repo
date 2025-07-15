"""
03_mmr_retriever.py
Example: Using Maximal Marginal Relevance (MMR) Retriever in LangChain
Step-by-step with comments. Uses HuggingFace open-source embeddings.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 2. Prepare some mock documents
docs = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="Paris is known for the Eiffel Tower."),
    Document(page_content="The capital of Germany is Berlin."),
    Document(page_content="Berlin has a rich history."),
    
]

# 3. Create a HuggingFace embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# 4. Create a FAISS vector store and add documents
vectorstore = FAISS.from_documents(docs, embeddings)

retriver = vectorstore.as_retriever(
    search_type="mmr",  # Use MMR for diverse results
    search_kwargs={"k": 2, "lambda_mult": 0.5}  # Retrieve top 2 results with diversity

)

query = "What is the capital of France?"

results = retriver.invoke(query)
# 6. Print the results
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content[:200]}...")  # Print first 200 characters of content
    print("\n")