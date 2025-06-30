"""
1_basic_text_loader.py
---------------------
This script demonstrates the most basic usage of LangChain's TextLoader to load a plain text file as a document.
"""

from langchain.document_loaders import TextLoader

# Path to the text file you want to load
file_path = "sample.txt"

# Create a TextLoader instance
loader = TextLoader(file_path)

# Load the document(s)
documents = loader.load()

# Print the loaded documents
for doc in documents:
    print("Loaded document content:")
    print(doc.page_content) 