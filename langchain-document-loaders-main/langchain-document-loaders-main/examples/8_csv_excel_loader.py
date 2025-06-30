"""
8_csv_excel_loader.py
--------------------
This script demonstrates loading data from a CSV file and using an LLM to answer questions about the data.
"""

from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key)
load_dotenv()

# Path to the CSV file
csv_path = "sample.csv"

# Load the CSV data
documents = CSVLoader(csv_path).load()

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0)

# Create a QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Ask a question about the data
question = "What insights can you provide from this data?"

# Run the QA chain
answer = qa_chain.run(input_documents=documents, question=question)

print("Question:", question)
print("Answer:", answer) 