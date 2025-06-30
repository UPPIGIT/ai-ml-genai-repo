"""
Example 12: Advanced Pipeline - CSV QA with Embeddings, FAISS, and LLM
This script demonstrates a pipeline: load a CSV, split rows, embed chunks, store in FAISS, and answer a user query using similarity search and an LLM.
"""
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Path to your CSV file (replace with your file path)
csv_path = "sample_data.csv"

# Load the CSV file (each row is a Document)
loader = CSVLoader(file_path=csv_path)
documents = loader.load()

# Split each row's content into chunks
splitter = CharacterTextSplitter(separator=" ", chunk_size=50, chunk_overlap=10)
all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

# Generate embeddings for each chunk
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(all_chunks, embeddings)

# Set up the LLM (using HuggingFaceHub; replace with your API key and model)
llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.1})

# Set up the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Example user query
query = "What is the main topic in the first few rows?"
result = qa.run(query)
print(f"Q: {query}\nA: {result}") 