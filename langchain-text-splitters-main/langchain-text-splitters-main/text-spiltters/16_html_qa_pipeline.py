"""
Example 16: HTML Retrieval QA Pipeline
This script demonstrates a retrieval-augmented QA pipeline for HTML: load HTML, split, embed, store in FAISS, and answer a user query using RetrievalQA and an LLM.
"""
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Path to your HTML file (replace with your file path)
html_path = "sample_page.html"

# Load the HTML document
loader = UnstructuredHTMLLoader(html_path)
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
query = "What is the main topic of this HTML page?"
result = qa.run(query)
print(f"Q: {query}\nA: {result}") 