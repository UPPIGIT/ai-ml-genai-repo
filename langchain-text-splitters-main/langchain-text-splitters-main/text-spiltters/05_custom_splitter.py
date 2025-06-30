"""
Example 5: Custom TextSplitter
This script demonstrates how to create a custom text splitter by subclassing LangChain's TextSplitter for special splitting logic (e.g., splitting on a custom delimiter).
"""
from langchain.text_splitter import TextSplitter

class CustomDelimiterSplitter(TextSplitter):
    def __init__(self, delimiter, chunk_size, chunk_overlap):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.delimiter = delimiter

    def split_text(self, text):
        # Split on the custom delimiter
        splits = text.split(self.delimiter)
        # Recombine splits into chunks of the desired size
        chunks = []
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) + len(self.delimiter) <= self._chunk_size:
                if current_chunk:
                    current_chunk += self.delimiter
                current_chunk += split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

# Example usage
text = "section1||section2||section3||section4"
splitter = CustomDelimiterSplitter(delimiter="||", chunk_size=20, chunk_overlap=0)
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}") 