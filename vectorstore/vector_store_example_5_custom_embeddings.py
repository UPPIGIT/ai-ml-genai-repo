# vector_store_example_5_custom_embeddings.py
"""
Step 5: Custom Embeddings with HuggingFace Model
- Uses a custom open-source model for embeddings
- Demonstrates integration with the vector store
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# 1. Load a custom HuggingFace model
custom_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(custom_model_name)

# 2. Example documents
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

# 3. Search function (same as before)
def search(query, top_k=2):
    query_emb = model.encode([query])[0]
    similarities = [
        np.dot(item["embedding"], query_emb) / (norm(item["embedding"]) * norm(query_emb))
        for item in vector_store
    ]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(vector_store[i]["text"], similarities[i]) for i in top_indices]

# 4. Example search
if __name__ == "__main__":
    query = "Animals outside"
    results = search(query)
    print(f"Query: {query}\nTop results with custom model:")
    for text, score in results:
        print(f"  - {text} (score: {score:.3f})") 