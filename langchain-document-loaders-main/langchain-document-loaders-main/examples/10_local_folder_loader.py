"""
10_local_folder_loader.py
------------------------
This script demonstrates loading all documents from a local folder, creating embeddings, and enabling semantic search with LLM-powered answers.
"""

import os
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv()

# Path to the folder containing documents
folder_path = "./documents"  # Place your documents here

# Load all text and PDF documents from the folder
loader = DirectoryLoader(
    folder_path,
    glob="**/*",
    loader_cls=lambda path: TextLoader(path) if path.endswith(".txt") else PyPDFLoader(path)
)
documents = loader.load()

# Create embeddings for the documents
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retriever and QA chain
retriever = vectorstore.as_retriever()
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Ask a question
question = "What information is available in this folder?"
answer = qa.run(question)

print("Question:", question)
print("Answer:", answer) 