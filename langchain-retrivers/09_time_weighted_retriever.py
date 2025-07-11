"""
09_time_weighted_retriever.py
Example: Using TimeWeightedVectorStoreRetriever in LangChain
Step-by-step with comments. Uses FAISS and HuggingFace embeddings.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.time_weighted_vector_store import TimeWeightedVectorStoreRetriever
import datetime

# 2. Prepare some mock documents with timestamps
docs = [
    Document(page_content="The capital of France is Paris.", metadata={"created_at": datetime.datetime(2024, 6, 1)}),
    Document(page_content="The tallest mountain is Mount Everest.", metadata={"created_at": datetime.datetime(2024, 6, 10)}),
    Document(page_content="Python is a popular programming language.", metadata={"created_at": datetime.datetime(2024, 6, 15)}),
]

# 3. Create a HuggingFace embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# 4. Create a FAISS vector store and add documents
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create the TimeWeightedVectorStoreRetriever
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,  # How quickly recency decays
    k=2,              # Number of results
    time_metadata_field="created_at"
)

# 6. Define a query
query = "What is the capital of France?"

# 7. Retrieve relevant documents (recent docs are prioritized)
results = retriever.get_relevant_documents(query)

# 8. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content} (created_at: {doc.metadata['created_at']})") 