"""
Example 2: RecursiveCharacterTextSplitter Usage
This script demonstrates how to use LangChain's RecursiveCharacterTextSplitter to split text more intelligently, respecting sentence and word boundaries.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample text (could be a long paragraph or document)
text = """
LangChain makes it easy to build LLM-powered applications. Sometimes, you need to split large documents into smaller pieces for processing, but you want to avoid breaking sentences or words awkwardly. RecursiveCharacterTextSplitter helps with this by trying different separators in order.
"""

# Initialize the recursive splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,         # Each chunk will have up to 60 characters
    chunk_overlap=15,      # Overlap 15 characters between chunks
    separators=["\n\n", "\n", ".", " ", ""]  # Try to split on paragraphs, then lines, then sentences, then spaces, then characters
)

# Split the text
chunks = splitter.split_text(text)

# Print the resulting chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n---") 