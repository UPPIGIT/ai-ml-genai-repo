# vector_store_example_2_update_delete.py
"""
Step 2: Update and Delete in Vector Store
- Builds on the basic example
- Shows how to update and delete documents
"""

from sentence_transformers import SentenceTransformer
import numpy as np
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

def search(query, top_k=2):
    query_emb = model.encode([query])[0]
    similarities = [
        np.dot(item["embedding"], query_emb) / (norm(item["embedding"]) * norm(query_emb))
        for item in vector_store
    ]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(vector_store[i]["text"], similarities[i]) for i in top_indices]

# 2. Update a document in the vector store
# Let's update the second document
new_text = "A dog is playing in the park."
new_embedding = model.encode([new_text])[0]
vector_store[1]["text"] = new_text
vector_store[1]["embedding"] = new_embedding

# 3. Delete a document from the vector store
# Let's delete the first document
vector_store.pop(0)

# 4. Example search after update and delete
if __name__ == "__main__":
    query = "Dog in the park"
    results = search(query)
    print(f"Query: {query}\nTop results after update and delete:")
    for text, score in results:
        print(f"  - {text} (score: {score:.3f})") 