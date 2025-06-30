"""
9_multi_page_web_scraper.py
--------------------------
This script demonstrates crawling multiple web pages, loading their content, and building a Q&A system over the entire site.
"""

from langchain.document_loaders import WebBaseLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key)
load_dotenv()

# List of URLs to crawl (replace with your own list)
urls = [
    "https://www.example.com",
    "https://www.example.com/about",
]

# Load all web pages
documents = []
for url in urls:
    try:
        documents.extend(WebBaseLoader(url).load())
    except Exception as e:
        print(f"Failed to load {url}: {e}")

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0)

# Create a QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Ask a question about the site
question = "What is this site about?"

# Run the QA chain
answer = qa_chain.run(input_documents=documents, question=question)

print("Question:", question)
print("Answer:", answer) 