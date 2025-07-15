from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Create a FAISS vector store
vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings,
  #  persist_directory="sample_faiss_db"  # Directory to store the vector store
)

similarity_search = vector_store.as_retriever(
    search_type="similarity",  # Use similarity search for related results
    search_kwargs={"k": 2}  # Retrieve top 2 results
)

# Initialize the MultiQueryRetriever with a ChatGroq model
multiquery_retriver = MultiQueryRetriever.from_llm(
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    retriever= vector_store.as_retriever(search_kwargs={"k": 2})
)
# Example query to retrieve information from the multi-query retriever
query = "How to improve energy levels and maintain balance?"


similarity_search_results = similarity_search.invoke(query)
for i, doc in enumerate(similarity_search_results):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters of content
    print("\n")

print("--" * 50)
multiquery_results = multiquery_retriver.invoke(query)
for i, doc in enumerate(multiquery_results):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters of content
    print("\n")