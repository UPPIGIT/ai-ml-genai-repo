"""
11_slack_loader.py
-----------------
This script demonstrates loading messages from a Slack channel and using an LLM to summarize discussions and generate meeting minutes.
"""

from langchain.document_loaders import SlackLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key and Slack credentials)
load_dotenv()

# Load messages from a Slack channel (requires Slack API token)
try:
    loader = SlackLoader()
    documents = loader.load()
except Exception as e:
    print("Slack loading failed (token required):", e)
    documents = []

if documents:
    # Initialize the OpenAI LLM
    llm = OpenAI(temperature=0)

    # Create a summarization chain
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")

    # Summarize the messages
    summary = summarize_chain.run(documents)
    print("Meeting minutes:", summary)
else:
    print("No messages loaded.") 