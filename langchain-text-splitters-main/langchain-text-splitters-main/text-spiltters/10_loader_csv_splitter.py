"""
Example 10: Loading a CSV File and Splitting Rows with CharacterTextSplitter
This script demonstrates how to load a CSV file using LangChain's CSVLoader and split each row's content using CharacterTextSplitter.
"""
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

# Path to your CSV file (replace with your file path)
csv_path = "sample_data.csv"

# Load the CSV file (each row is a Document)
loader = CSVLoader(file_path=csv_path)
documents = loader.load()

# Initialize the text splitter
splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=50,
    chunk_overlap=10
)

# Split each row's content into chunks
for row_num, doc in enumerate(documents, 1):
    chunks = splitter.split_text(doc.page_content)
    print(f"CSV Row {row_num} (source: {doc.metadata.get('source', 'N/A')}):")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n---") 