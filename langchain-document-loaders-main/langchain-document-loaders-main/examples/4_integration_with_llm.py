"""
4_integration_with_llm.py
------------------------
This script demonstrates loading a document and using an LLM (OpenAI) to answer a question about its content.
"""

import os
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv()

# Load a document (using the sample.txt from previous example)
loader = TextLoader("sample.txt")
documents = loader.load()

# Initialize the OpenAI LLM (ensure OPENAI_API_KEY is set in your .env file)
llm = OpenAI(temperature=0)

# Create a QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Ask a question about the document
question = "What is this document about?"

# Run the QA chain
answer = qa_chain.run(input_documents=documents, question=question)

print("Question:", question)
print("Answer:", answer) 