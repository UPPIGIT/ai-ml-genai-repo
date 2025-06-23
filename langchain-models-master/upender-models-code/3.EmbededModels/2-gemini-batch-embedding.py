from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", dimensions=20)
documents = ["hello, world!", "this is a test", "langchain is great", "I love langchain"]
vectors = embeddings.embed_documents(documents)
print(vectors)