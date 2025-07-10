# vector_store_example_3_persistence.py
"""
Step 3: Persistence (Save/Load) in Vector Store
- Shows how to save the vector store to disk and load it back
- Uses Python's pickle module for serialization
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from numpy.linalg import norm

# 1. Load model and initialize documents
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "The cat sits outside.",
    "A dog barks loudly.",
    "Birds are flying in the sky.",
    "The sun is bright today."
]
embeddings = model.encode(documents)
vector_store = [
    {"text": doc, "embedding": emb}
    for doc, emb in zip(documents, embeddings)
]

# 2. Save the vector store to disk
def save_vector_store(filename):
    with open(filename, 'wb') as f:
        pickle.dump(vector_store, f)

# 3. Load the vector store from disk
def load_vector_store(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 4. Search function (same as before)
def search(query, top_k=2, store=None):
    if store is None:
        store = vector_store
    query_emb = model.encode([query])[0]
    similarities = [
        np.dot(item["embedding"], query_emb) / (norm(item["embedding"]) * norm(query_emb))
        for item in store
    ]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(store[i]["text"], similarities[i]) for i in top_indices]

# 5. Example usage
if __name__ == "__main__":
    # Save the current vector store
    save_vector_store('vector_store.pkl')
    print("Vector store saved to 'vector_store.pkl'.")

    # Load the vector store back
    loaded_store = load_vector_store('vector_store.pkl')
    print("Vector store loaded from 'vector_store.pkl'.")

    # Search using the loaded store
    query = "Animals outside"
    results = search(query, store=loaded_store)
    print(f"Query: {query}\nTop results from loaded store:")
    for text, score in results:
        print(f"  - {text} (score: {score:.3f})") 