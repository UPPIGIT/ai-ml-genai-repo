"""
7_gmail_loader.py
-----------------
This script demonstrates loading emails from Gmail and using an LLM to summarize and extract action items.
"""

from langchain.document_loaders import GmailLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key and Gmail credentials)
load_dotenv()

# Load emails (requires Gmail API setup and credentials.json)
try:
    loader = GmailLoader()
    documents = loader.load()
except Exception as e:
    print("Gmail loading failed (credentials required):", e)
    documents = []

if documents:
    # Initialize the OpenAI LLM
    llm = OpenAI(temperature=0)

    # Create a summarization chain
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")

    # Summarize the emails
    summary = summarize_chain.run(documents)
    print("Summary of recent emails:", summary)
else:
    print("No emails loaded.") 