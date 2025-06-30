"""
Example 14: DOCX Pipeline - Load, Split, and Embed
This script demonstrates loading a DOCX file, splitting it into chunks, generating embeddings, and printing embedding info.
"""
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Path to your DOCX file (replace with your file path)
docx_path = "sample_document.docx"

# Load the DOCX document
loader = UnstructuredWordDocumentLoader(docx_path)
documents = loader.load()

# Split each document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

# Generate embeddings for each chunk
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = embeddings.embed_documents(all_chunks)

print(f"Total chunks: {len(all_chunks)}")
print(f"Embedding shape for first chunk: {len(chunk_embeddings[0])} dimensions") 