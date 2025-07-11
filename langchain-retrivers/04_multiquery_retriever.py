"""
04_multiquery_retriever.py
Example: Using MultiQueryRetriever in LangChain
Step-by-step with comments. Uses HuggingFace open-source embeddings and LLM.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever
from transformers import pipeline

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

# 5. Create a HuggingFace LLM pipeline (e.g., flan-tiny for demo)
generator = pipeline("text-generation", model="google/flan-tiny")
llm = HuggingFacePipeline(pipeline=generator)

# 6. Create a MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 7. Retrieve documents for a broad query
query = "Tell me about European capitals."
results = multiquery_retriever.get_relevant_documents(query)

# 8. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}") 