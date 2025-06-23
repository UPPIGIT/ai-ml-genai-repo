from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Change to "cuda" if you have a GPU
)
documents = ["hello, world!", "this is a test", "langchain is great", "I love langchain"]
vectors = embeddings.embed_documents(documents)
print(str(vectors))  # Print first 100 characters of the first vector