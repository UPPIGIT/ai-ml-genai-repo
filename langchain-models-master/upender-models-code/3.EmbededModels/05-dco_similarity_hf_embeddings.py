from langchain_huggingface import HuggingFaceEmbeddings
"""
This script demonstrates how to use HuggingFace sentence-transformer embeddings with LangChain
to compute semantic similarity between a query and a set of documents using cosine similarity.

Steps performed:
1. Initializes a HuggingFace embedding model ('sentence-transformers/all-MiniLM-L6-v2').
2. Defines a list of sample cricket-related documents.
3. Generates embeddings for each document.
4. Embeds a query about the best finisher in cricket.
5. Computes cosine similarity scores between the query embedding and each document embedding.
6. Prints the similarity scores and identifies the most similar document to the query.

Dependencies:
- langchain_huggingface
- scikit-learn
- numpy
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample documents
documents = [
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Virat Kohli is known for his aggressive batting style and chase mastery.",
    "Sachin Tendulkar is considered the god of cricket in India."
]

# Create embeddings for documents
doc_embeddings = embeddings.embed_documents(documents)

# Query text
query = "Who is the best finisher in cricket?"
query_embedding = embeddings.embed_query(query)

# Calculate similarity scores
similarity_scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Print results
print(f"Similarity Scores: {similarity_scores}")

# Find most similar document
most_similar_idx = np.argmax(similarity_scores)
print(f"\nDocument: {documents[most_similar_idx]}")
print(f"Similarity Score: {similarity_scores[most_similar_idx]}")