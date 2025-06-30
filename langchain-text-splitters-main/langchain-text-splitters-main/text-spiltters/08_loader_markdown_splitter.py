"""
Example 8: Loading a Markdown File and Splitting with MarkdownHeaderTextSplitter
This script demonstrates how to load a Markdown file using LangChain's TextLoader and split it into sections using MarkdownHeaderTextSplitter.
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Path to your markdown file (replace with your file path)
file_path = "sample_markdown.md"

# Load the document
loader = TextLoader(file_path)
documents = loader.load()

# Define which headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Split each document into sections
for doc in documents:
    sections = splitter.split_text(doc.page_content)
    print(f"Document: {doc.metadata.get('source', 'N/A')}")
    for i, section in enumerate(sections):
        print(f"Section {i+1}: {section['header']}")
        print(section['content'])
        print("---") 