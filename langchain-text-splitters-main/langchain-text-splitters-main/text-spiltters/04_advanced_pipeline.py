"""
Example 4: Real-World Pipeline - File Loading, Splitting, and Preparing for Embedding
This script demonstrates a typical workflow in a real project: loading a text file, splitting it into chunks, and preparing those chunks for embedding with an LLM.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Path to your text file (replace with your file path)
file_path = "sample_document.txt"

# Load the document
loader = TextLoader(file_path)
documents = loader.load()

# Use RecursiveCharacterTextSplitter for robust splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

# Split all documents into chunks
all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

# Now, all_chunks can be sent to an embedding model or LLM
print(f"Total chunks: {len(all_chunks)}")
for i, chunk in enumerate(all_chunks[:3]):  # Show first 3 chunks
    print(f"Chunk {i+1}:\n{chunk}\n---") 