"""
Example 7: Loading a Text File and Splitting with RecursiveCharacterTextSplitter
This script demonstrates how to load a text file using LangChain's TextLoader and split it into chunks using RecursiveCharacterTextSplitter for more natural chunking.
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your text file (replace with your file path)
file_path = "sample_document.txt"

# Load the document
loader = TextLoader(file_path)
documents = loader.load()

# Initialize the recursive text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Split each document into chunks
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    print(f"Document: {doc.metadata.get('source', 'N/A')}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n---") 