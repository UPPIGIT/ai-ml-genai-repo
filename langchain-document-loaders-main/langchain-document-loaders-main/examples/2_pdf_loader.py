"""
2_pdf_loader.py
--------------
This script demonstrates how to use LangChain's PyPDFLoader to load a PDF file as documents.
"""

from langchain.document_loaders import PyPDFLoader

# Path to the PDF file you want to load
file_path = "sample.pdf"

# Create a PyPDFLoader instance
loader = PyPDFLoader(file_path)

# Load the document(s)
documents = loader.load()

# Print the loaded documents
for i, doc in enumerate(documents):
    print(f"--- Page {i+1} ---")
    print(doc.page_content) 