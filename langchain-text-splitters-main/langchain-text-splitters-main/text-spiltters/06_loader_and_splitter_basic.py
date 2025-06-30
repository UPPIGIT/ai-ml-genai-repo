"""
Example 6: Loading a Text File and Splitting with CharacterTextSplitter
This script demonstrates how to load a text file using LangChain's TextLoader and split it into chunks using CharacterTextSplitter.
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Path to your text file (replace with your file path)
file_path = "sample_document.txt"

# Load the document
loader = TextLoader(file_path)
documents = loader.load()

# Initialize the text splitter
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
)

# Split each document into chunks
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    print(f"Document: {doc.metadata.get('source', 'N/A')}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n---") 