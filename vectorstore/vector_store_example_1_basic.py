# vector_store_example_1_basic.py
"""
Step 1: Basic Vector Store Example
- Uses SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- Adds documents to a simple in-memory vector store
- Performs a similarity search
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load an open-source embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Example documents
documents = [
    "The cat sits outside.",
    "A dog barks loudly.",
    "Birds are flying in the sky.",
    "The sun is bright today."
]

# 3. Generate embeddings for each document
embeddings = model.encode(documents)

# 4. Simple in-memory vector store (list of dicts)
vector_store = [
    {"text": doc, "embedding": emb}
    for doc, emb in zip(documents, embeddings)
]

# 5. Function to perform a similarity search
from numpy.linalg import norm

def search(query, top_k=2):
    query_emb = model.encode([query])[0]
    # Compute cosine similarity
    similarities = [
        np.dot(item["embedding"], query_emb) / (norm(item["embedding"]) * norm(query_emb))
        for item in vector_store
    ]
    # Get top_k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(vector_store[i]["text"], similarities[i]) for i in top_indices]

# 6. Example search
if __name__ == "__main__":
    query = "Animals outside"
    results = search(query)
    print(f"Query: {query}\nTop results:")
    for text, score in results:
        print(f"  - {text} (score: {score:.3f})") 