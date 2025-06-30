"""
Example 11: Advanced Pipeline - PDF to Embedding with LLM Integration
This script demonstrates a real-world pipeline: loading a PDF, splitting it into chunks, generating embeddings for each chunk using a HuggingFace LLM, and printing the embedding shape.
"""
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Path to your PDF file (replace with your file path)
pdf_path = "sample_document.pdf"

# Load the PDF document
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split each page into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

# Initialize HuggingFace embeddings (using a small model for demo; replace as needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings for each chunk
chunk_embeddings = embeddings.embed_documents(all_chunks)

print(f"Total chunks: {len(all_chunks)}")
print(f"Embedding shape for first chunk: {len(chunk_embeddings[0])} dimensions") 