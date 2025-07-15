from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainExtractor

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Load environment variables

load_dotenv()

# Initialize the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Create a FAISS vector store
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
    # persist_directory="sample_faiss_db"  # Directory to store the vector store

)

# Initialize the Document Compressor
compressor = LLMChainExtractor.from_llm(
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    #max_length=100  # Maximum length of the compressed document
)
# Create the Contextual Compression Retriever
retriever = ContextualCompressionRetriever(
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    base_compressor=compressor,
)
# Example query to retrieve information from the contextual compression retriever
query = "What is the significance of photosynthesis in plants?"
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters of content
    print("\n")