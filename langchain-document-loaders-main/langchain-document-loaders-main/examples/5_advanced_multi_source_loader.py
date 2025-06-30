"""
5_advanced_multi_source_loader.py
-------------------------------
This script demonstrates loading documents from multiple sources (text, PDF, web), combining them, and using an LLM to answer a question about all sources.
"""

import os
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv()

# Load documents from multiple sources
text_docs = TextLoader("sample.txt").load()
pdf_docs = []
try:
    pdf_docs = PyPDFLoader("sample.pdf").load()
except Exception as e:
    print("PDF loading failed (sample.pdf missing?):", e)
web_docs = []
try:
    web_docs = WebBaseLoader("https://www.example.com").load()
except Exception as e:
    print("Web loading failed:", e)

# Combine all documents
all_docs = text_docs + pdf_docs + web_docs

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0)

# Create a QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Ask a question about the combined documents
question = "Summarize the content from all sources."

# Run the QA chain
answer = qa_chain.run(input_documents=all_docs, question=question)

print("Question:", question)
print("Answer:", answer) 