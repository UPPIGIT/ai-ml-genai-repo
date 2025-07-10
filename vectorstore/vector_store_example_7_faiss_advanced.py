# vector_store_example_7_faiss_advanced.py
"""
Step 7: Advanced FAISS Usage
- Uses FAISS IndexIVFFlat for fast, scalable search
- Shows how to train, add, search, and save/load the FAISS index
- Includes error handling for index loading
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# 1. Load model and create documents
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [f"Document number {i}" for i in range(1000)]
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# 2. Set up FAISS IndexIVFFlat (inverted file index)
embedding_dim = embeddings.shape[1]
nlist = 50  # Number of clusters
quantizer = faiss.IndexFlatL2(embedding_dim)
index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

# 3. Train the index (required for IVFFlat)
if not index.is_trained:
    index.train(embeddings)

# 4. Add embeddings to the index
index.add(embeddings)

# 5. Save the index to disk
def save_faiss_index(index, filename):
    faiss.write_index(index, filename)
    print(f"FAISS index saved to {filename}")

# 6. Load the index from disk
def load_faiss_index(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Index file {filename} not found.")
    return faiss.read_index(filename)

# 7. Search function using FAISS
def search(query, top_k=5, idx=index):
    query_emb = model.encode([query]).astype('float32')
    D, I = idx.search(query_emb, top_k)
    results = [(documents[i], D[0][idx]) for idx, i in enumerate(I[0])]
    return results

# 8. Example usage
if __name__ == "__main__":
    # Save the index
    save_faiss_index(index, 'faiss_ivfflat.index')

    # Load the index (with error handling)
    try:
        loaded_index = load_faiss_index('faiss_ivfflat.index')
        print("FAISS index loaded from disk.")
    except FileNotFoundError as e:
        print(e)
        loaded_index = index  # fallback

    # Search using the loaded index
    query = "Document number 10"
    results = search(query, idx=loaded_index)
    print(f"Query: {query}\nTop results from FAISS IVFFlat:")
    for text, dist in results:
        print(f"  - {text} (distance: {dist:.3f})") 