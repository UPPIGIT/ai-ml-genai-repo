from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07",dimensions=20)
#embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",dimensions=20)
vector = embeddings.embed_query("hello, world!")
print(vector)