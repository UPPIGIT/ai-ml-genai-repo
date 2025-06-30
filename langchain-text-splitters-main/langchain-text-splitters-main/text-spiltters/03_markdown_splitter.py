"""
Example 3: MarkdownHeaderTextSplitter Usage
This script demonstrates how to split a Markdown document into sections based on headers using LangChain's MarkdownHeaderTextSplitter.
"""
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Sample Markdown text
markdown_text = """
# Project Overview
LangChain is a powerful framework for LLM applications.

## Features
- Text splitting
- Document loading

## Usage
You can use LangChain for many NLP tasks.
"""

# Define which headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Split the markdown text
sections = splitter.split_text(markdown_text)

# Print the resulting sections
for i, section in enumerate(sections):
    print(f"Section {i+1}: {section['header']}")
    print(section['content'])
    print("---") 