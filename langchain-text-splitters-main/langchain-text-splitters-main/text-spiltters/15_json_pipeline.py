"""
Example 15: JSON Pipeline - Load, Split, and Embed
This script demonstrates loading a JSON file, splitting it into chunks, generating embeddings, and printing embedding info.
"""
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Path to your JSON file (replace with your file path)
json_path = "sample_data.json"

# Load the JSON document (custom loader usage; adjust jq_schema as needed)
loader = JSONLoader(
    file_path=json_path,
    jq_schema=".[] | .text",  # Assumes each item has a 'text' field
    text_content=False
)
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