"""
07_self_query_retriever.py
Example: Using SelfQueryRetriever in LangChain
Step-by-step with comments. Uses HuggingFace LLM and open-source embeddings.
"""

# 1. Import necessary modules
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# 2. Prepare some mock documents with metadata
docs = [
    Document(page_content="The capital of France is Paris.", metadata={"country": "France", "type": "capital"}),
    Document(page_content="The capital of Germany is Berlin.", metadata={"country": "Germany", "type": "capital"}),
    Document(page_content="Mount Everest is the tallest mountain.", metadata={"country": "Nepal", "type": "mountain"}),
]

# 3. Create a HuggingFace embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# 4. Create a FAISS vector store and add documents
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create a HuggingFace LLM pipeline (e.g., flan-tiny for demo)
generator = pipeline("text-generation", model="google/flan-tiny")
llm = HuggingFacePipeline(pipeline=generator)

# 6. Define a prompt template for the LLM to generate filters
prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query: '{query}', generate a filter for the metadata."
)

# 7. Create the SelfQueryRetriever
self_query_retriever = SelfQueryRetriever(
    vectorstore=vectorstore,
    llm=llm,
    document_content_description="A collection of world facts.",
    metadata_field_info=[
        {"name": "country", "description": "The country related to the fact."},
        {"name": "type", "description": "The type of fact (capital, mountain, etc.)."}
    ],
    prompt=prompt,
)

# 8. Define a query with a filter
query = "What is the capital of Germany?"

# 9. Retrieve relevant documents
results = self_query_retriever.get_relevant_documents(query)

# 10. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content} (metadata: {doc.metadata})") 