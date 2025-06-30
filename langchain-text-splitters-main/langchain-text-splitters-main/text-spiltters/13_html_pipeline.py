"""
Example 13: HTML Pipeline - Load, Split, and Embed
This script demonstrates loading an HTML file, splitting it into chunks, generating embeddings, and printing embedding info.
"""
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Path to your HTML file (replace with your file path)
html_path = "sample_page.html"

# Load the HTML document
loader = UnstructuredHTMLLoader(html_path)
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