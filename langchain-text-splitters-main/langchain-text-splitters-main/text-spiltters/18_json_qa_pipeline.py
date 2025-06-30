"""
Example 18: JSON Retrieval QA Pipeline
This script demonstrates a retrieval-augmented QA pipeline for JSON: load JSON, split, embed, store in FAISS, and answer a user query using RetrievalQA and an LLM.
"""
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Path to your JSON file (replace with your file path)
json_path = "sample_data.json"

# Load the JSON document (custom loader usage; adjust jq_schema as needed)
loader = JSONLoader(
    file_path=json_path,
    jq_schema=".[] | .text",  # Assumes each item has a 'text' field
    text_content=False
)
documents = loader.load()

# Split each document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

# Generate embeddings and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(all_chunks, embeddings)

# Set up the LLM (using HuggingFaceHub; replace with your API key and model)
llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.1})

# Set up the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Example user query
query = "What are the key facts in this JSON data?"
result = qa.run(query)
print(f"Q: {query}\nA: {result}") 