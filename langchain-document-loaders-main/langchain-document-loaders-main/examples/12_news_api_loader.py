"""
12_news_api_loader.py
--------------------
This script demonstrates loading news articles from an API, summarizing trends, and answering questions about recent events using an LLM.
"""

import requests
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key and News API key)
load_dotenv()

# Example: Load news articles from NewsAPI.org (replace with your API key)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

response = requests.get(url)
articles = response.json().get("articles", [])

# Combine all article descriptions into one document
news_content = "\n".join([a["title"] + ": " + (a["description"] or "") for a in articles])

# Save to a temporary file for loading
with open("news_temp.txt", "w", encoding="utf-8") as f:
    f.write(news_content)

# Load the news content as a document
documents = TextLoader("news_temp.txt").load()

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0)

# Create a QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Ask a question about the news
question = "What are the main trends in today's news?"

# Run the QA chain
answer = qa_chain.run(input_documents=documents, question=question)

print("Question:", question)
print("Answer:", answer) 