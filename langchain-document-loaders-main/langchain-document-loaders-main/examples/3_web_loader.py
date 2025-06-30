"""
3_web_loader.py
--------------
This script demonstrates how to use LangChain's WebBaseLoader to load content from a web page as a document.
"""

from langchain.document_loaders import WebBaseLoader

# URL of the web page you want to load
url = "https://www.example.com"

# Create a WebBaseLoader instance
loader = WebBaseLoader(url)

# Load the document(s)
documents = loader.load()

# Print the loaded documents
for doc in documents:
    print("Loaded web page content:")
    print(doc.page_content[:500])  # Print only the first 500 characters 