"""
06_bm25_retriever.py
Example: Using BM25Retriever in LangChain
Step-by-step with comments. Uses only open-source components.
"""

# 1. Import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 2. Prepare some mock documents
docs = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="The tallest mountain is Mount Everest."),
    Document(page_content="Python is a popular programming language."),
]

# 3. Create the BM25Retriever
bm25_retriever = BM25Retriever.from_documents(docs)

# 4. Define a query
query = "What is the capital of France?"

# 5. Retrieve relevant documents
results = bm25_retriever.invoke(query)

# 6. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}") 