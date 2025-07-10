# vector_store_example_4_advanced_search.py
"""
Step 4: Advanced Search in Vector Store
- Adds metadata to documents
- Demonstrates filtering by metadata and batch queries
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# 1. Load model and initialize documents with metadata
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    {"text": "The cat sits outside.", "type": "animal", "id": 1},
    {"text": "A dog barks loudly.", "type": "animal", "id": 2},
    {"text": "Birds are flying in the sky.", "type": "animal", "id": 3},
    {"text": "The sun is bright today.", "type": "weather", "id": 4}
]
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)
vector_store = [
    {"text": doc["text"], "embedding": emb, "type": doc["type"], "id": doc["id"]}
    for doc, emb in zip(documents, embeddings)
]

# 2. Search with optional metadata filter
def search(query, top_k=2, filter_type=None):
    query_emb = model.encode([query])[0]
    filtered_store = [item for item in vector_store if (filter_type is None or item["type"] == filter_type)]
    if not filtered_store:
        return []
    similarities = [
        np.dot(item["embedding"], query_emb) / (norm(item["embedding"]) * norm(query_emb))
        for item in filtered_store
    ]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(filtered_store[i]["text"], similarities[i], filtered_store[i]["type"]) for i in top_indices]

# 3. Batch query search
def batch_search(queries, top_k=2):
    results = {}
    for query in queries:
        results[query] = search(query, top_k=top_k)
    return results

# 4. Example usage
if __name__ == "__main__":
    # Search with metadata filter
    query = "Animals outside"
    results = search(query, filter_type="animal")
    print(f"Query: {query}\nTop animal results:")
    for text, score, typ in results:
        print(f"  - {text} (type: {typ}, score: {score:.3f})")

    # Batch search
    queries = ["Animals outside", "Weather today"]
    batch_results = batch_search(queries)
    print("\nBatch search results:")
    for q, res in batch_results.items():
        print(f"Query: {q}")
        for text, score, typ in res:
            print(f"  - {text} (type: {typ}, score: {score:.3f})") 