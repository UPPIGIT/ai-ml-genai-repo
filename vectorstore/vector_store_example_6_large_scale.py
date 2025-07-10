# vector_store_example_6_large_scale.py
"""
Step 6: Large-Scale Vector Search with FAISS
- Uses FAISS for efficient similarity search on large datasets
- Integrates with SentenceTransformers for embeddings
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 1. Load model and create a large set of documents
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [f"Document number {i}" for i in range(1000)]  # Example: 1000 documents
embeddings = model.encode(documents, show_progress_bar=True)

# 2. Build FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (can use IndexFlatIP for cosine similarity)
index.add(np.array(embeddings).astype('float32'))

# 3. Search function using FAISS
def search(query, top_k=5):
    query_emb = model.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k)
    results = [(documents[i], D[0][idx]) for idx, i in enumerate(I[0])]
    return results

# 4. Example search
if __name__ == "__main__":
    query = "Document number 10"
    results = search(query)
    print(f"Query: {query}\nTop results from FAISS:")
    for text, dist in results:
        print(f"  - {text} (distance: {dist:.3f})") 