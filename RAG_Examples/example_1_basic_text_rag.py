from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
# Step 1: Prepare sample documents
documents = [
        Document(page_content="Python is a high-level programming language known for its simplicity and readability.", 
                metadata={"source": "python_intro.txt"}),
        Document(page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.", 
                metadata={"source": "ml_basics.txt"}),
        Document(page_content="RAG (Retrieval-Augmented Generation) combines retrieval and generation for better AI responses.", 
                metadata={"source": "rag_explanation.txt"}),
        Document(page_content="LangChain is a framework for developing applications powered by language models.", 
                metadata={"source": "langchain_info.txt"}),
        Document(page_content="Vector databases store high-dimensional vectors for efficient similarity search.", 
                metadata={"source": "vector_db.txt"})
    ]

# Step 2: Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20 , 
    length_function=len
    )

spilts = text_splitter.split_documents(documents)
# Step 3: Print the resulting chunks
for i, chunk in enumerate(spilts):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print(f"Metadata: {chunk.metadata}")
    print("-" * 40)

# Step 4: Create embeddings for the chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Create a vector store using Chroma
vector_store = Chroma.from_documents(
    documents=spilts, 
    embedding=embeddings, 
    persist_directory="basic_rag_chroma_db"
)   

# Step 6: Print the number of documents in the vector store
print(f"Number of documents in the vector store: {len(vector_store)}")

prompt_template = """
    Use the following context to answer the question. If you don't know the answer based on the context, say No.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

#groq model
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)
# Step 7: Create a RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Using embeddings as a placeholder for the LLM
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Step 8: Ask a question and get the answer
question = "What is RAG in the context of AI?"
result = rag_chain.invoke({"query": question})
# Step 9: Print the result
print("Question:", question)
print("Answer:", result['result'])
print(f"Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")
print("-" * 40)
question = "What is google cloud"
result = rag_chain.invoke({"query": question})
# Print the result
print("Question:", question)
print("Answer:", result['result'])
print(f"Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")

# Step 10: Clean up the vector store
vector_store.delete_collection()
# Note: The vector store is deleted to clean up resources after the example.


    