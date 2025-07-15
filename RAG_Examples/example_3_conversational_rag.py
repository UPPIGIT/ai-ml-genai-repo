import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()

"""
    Conversational RAG with memory to maintain context across interactions.
    Demonstrates multi-turn conversations with document retrieval.
    """

print("\n=== Example 3: Conversational RAG with Memory ===")

  # Step 1: Create knowledge base
knowledge_base = [
        Document(page_content="Climate change refers to long-term shifts in temperatures and weather patterns.", 
                metadata={"topic": "climate", "source": "climate_basics.txt"}),
        Document(page_content="Renewable energy sources include solar, wind, hydro, and geothermal power.", 
                metadata={"topic": "energy", "source": "renewable_energy.txt"}),
        Document(page_content="Carbon footprint is the total amount of greenhouse gases produced by human activities.", 
                metadata={"topic": "environment", "source": "carbon_footprint.txt"}),
        Document(page_content="Sustainable development meets present needs without compromising future generations.", 
                metadata={"topic": "sustainability", "source": "sustainability.txt"}),
        Document(page_content="Electric vehicles produce zero direct emissions and are becoming more popular.", 
                metadata={"topic": "transportation", "source": "electric_vehicles.txt"})
    ]

    # Step 2: Split and create vector store
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )


splits = text_splitter.split_documents(knowledge_base)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="conversational_rag_chroma_db"
)

# Step 3: Create retriever
retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)

# Step 4: Create Conversational Retrieval Chain
conversational_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)

   # Step 6: Simulate conversation
conversation_queries = [
        "What is climate change?",
        "How can renewable energy help with this issue?",
        "What about electric vehicles? Are they related to what we discussed?",
        "Can you summarize our conversation so far?"
    ]


print("\nStarting conversational RAG session:")

for i, query in enumerate(conversation_queries, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {query}")
        
        result = conversational_rag_chain.invoke({"question": query})
        print(f"Assistant: {result['answer']}")
        
        # Show source documents for transparency
        if result.get('source_documents'):
            sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
            print(f"Sources: {list(set(sources))}")