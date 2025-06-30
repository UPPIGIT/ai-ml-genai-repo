"""
Example 9: Loading a PDF File and Splitting with RecursiveCharacterTextSplitter
This script demonstrates how to load a PDF file using LangChain's PyPDFLoader and split it into chunks using RecursiveCharacterTextSplitter.
"""
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your PDF file (replace with your file path)
pdf_path = "sample_document.pdf"

# Load the PDF document (each page is a Document)
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Initialize the recursive text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# Split each page into chunks
for page_num, doc in enumerate(documents, 1):
    chunks = splitter.split_text(doc.page_content)
    print(f"PDF Page {page_num} (source: {doc.metadata.get('source', 'N/A')}):")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n---") 