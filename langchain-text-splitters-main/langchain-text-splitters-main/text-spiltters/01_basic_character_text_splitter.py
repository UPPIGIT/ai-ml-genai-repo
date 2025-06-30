"""
Example 1: Basic CharacterTextSplitter Usage
This script demonstrates how to use LangChain's CharacterTextSplitter to split a long text into smaller chunks based on character count.
"""
from langchain.text_splitter import CharacterTextSplitter

# Sample text (could be a document, article, etc.)
text = """
LangChain is a framework for developing applications powered by language models. It enables easy chaining of LLMs with other sources of computation or knowledge.
This example shows how to split this text into smaller chunks for easier processing.
"""

# Initialize the splitter
splitter = CharacterTextSplitter(
    separator="\n",  # Split on newlines
    chunk_size=50,     # Each chunk will have up to 50 characters
    chunk_overlap=10   # Overlap 10 characters between chunks
)

# Split the text
chunks = splitter.split_text(text)

# Print the resulting chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n---") 