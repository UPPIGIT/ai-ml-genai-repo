# Complete Guide to Vector Store Methods and Actions in LangChain

## Table of Contents
1. [Basic CRUD Operations](#basic-crud-operations)
2. [Search and Retrieval Methods](#search-and-retrieval-methods)
3. [Document Management](#document-management)
4. [Indexing and Optimization](#indexing-and-optimization)
5. [Metadata Operations](#metadata-operations)
6. [Persistence and Storage](#persistence-and-storage)
7. [Advanced Operations](#advanced-operations)
8. [Vector Store Specific Methods](#vector-store-specific-methods)

## 1. Basic CRUD Operations

### Create Operations

```python
from langchain_community.vectorstores import FAISS, Chroma, Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import numpy as np

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Method 1: Create from documents
documents = [
    Document(page_content="Python is a programming language", metadata={"type": "tech"}),
    Document(page_content="The sky is blue", metadata={"type": "nature"})
]

# Create vector store from documents
vector_store = FAISS.from_documents(documents, embeddings)

# Method 2: Create from texts
texts = ["Hello world", "Machine learning is amazing"]
metadatas = [{"source": "greeting"}, {"source": "ml"}]
vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Method 3: Create empty and add later
vector_store = FAISS(embeddings.embed_query, None, embeddings, {})

# Method 4: Create from existing embeddings
texts = ["Sample text 1", "Sample text 2"]
embeddings_list = [embeddings.embed_query(text) for text in texts]
vector_store = FAISS.from_embeddings(
    text_embeddings=list(zip(texts, embeddings_list)),
    embedding=embeddings,
    metadatas=metadatas
)
```

### Read Operations

```python
# Get document by ID (if supported)
try:
    doc = vector_store.get(doc_id="some_id")
except:
    print("Get by ID not supported by this vector store")

# Get all documents (if supported)
try:
    all_docs = vector_store.get()
except:
    print("Get all not supported by this vector store")

# Check if document exists
def document_exists(vector_store, query_text, threshold=0.9):
    results = vector_store.similarity_search_with_score(query_text, k=1)
    if results and results[0][1] > threshold:
        return True
    return False
```

### Update Operations

```python
# Update document content (varies by vector store)
def update_document(vector_store, doc_id, new_content, new_metadata=None):
    try:
        # For stores that support direct updates
        vector_store.update_document(doc_id, new_content, new_metadata)
    except:
        # For stores that don't support updates, delete and re-add
        vector_store.delete([doc_id])
        vector_store.add_texts([new_content], metadatas=[new_metadata])

# Update metadata only
def update_metadata(vector_store, doc_id, new_metadata):
    try:
        vector_store.update_metadata(doc_id, new_metadata)
    except:
        print("Metadata update not supported")
```

### Delete Operations

```python
# Delete by ID
vector_store.delete(["doc_id_1", "doc_id_2"])

# Delete by filter (for supported stores)
vector_store.delete(filter={"source": "old_data"})

# Delete all documents
try:
    vector_store.delete()  # Delete all
except:
    print("Delete all not supported")

# Conditional delete
def delete_by_similarity(vector_store, query, threshold=0.95):
    results = vector_store.similarity_search_with_score(query, k=100)
    ids_to_delete = [result[0].metadata.get('id') for result in results 
                     if result[1] > threshold]
    if ids_to_delete:
        vector_store.delete(ids_to_delete)
```

## 2. Search and Retrieval Methods

### Similarity Search

```python
# Basic similarity search
results = vector_store.similarity_search(
    query="machine learning",
    k=5  # Number of results
)

# Similarity search with scores
results_with_scores = vector_store.similarity_search_with_score(
    query="machine learning",
    k=5
)

for doc, score in results_with_scores:
    print(f"Score: {score:.3f}, Content: {doc.page_content[:100]}")

# Similarity search with relevance scores (0-1 normalized)
results_with_relevance = vector_store.similarity_search_with_relevance_scores(
    query="machine learning",
    k=5
)
```

### Advanced Search Methods

```python
# Maximum Marginal Relevance (MMR) search - reduces redundancy
mmr_results = vector_store.max_marginal_relevance_search(
    query="machine learning",
    k=5,
    fetch_k=20,  # Fetch more candidates
    lambda_mult=0.5  # Diversity parameter (0=max diversity, 1=max relevance)
)

# Search by vector
query_vector = embeddings.embed_query("machine learning")
vector_results = vector_store.similarity_search_by_vector(
    embedding=query_vector,
    k=5
)

# Search with threshold
def search_with_threshold(vector_store, query, threshold=0.8):
    results = vector_store.similarity_search_with_score(query, k=50)
    return [doc for doc, score in results if score >= threshold]
```

### Filtered Search

```python
# Search with metadata filters
filtered_results = vector_store.similarity_search(
    query="programming",
    k=5,
    filter={"type": "tech"}  # Only documents with type="tech"
)

# Complex filters (varies by vector store)
complex_filter = {
    "and": [
        {"type": {"$eq": "tech"}},
        {"date": {"$gte": "2023-01-01"}}
    ]
}

# Range-based filters
range_filter = {
    "score": {"$gte": 0.8, "$lte": 1.0}
}
```

## 3. Document Management

### Adding Documents

```python
# Add single document
new_doc = Document(
    page_content="New content",
    metadata={"type": "update", "timestamp": "2024-01-01"}
)
vector_store.add_documents([new_doc])

# Add multiple documents
new_docs = [
    Document(page_content="Doc 1", metadata={"batch": "1"}),
    Document(page_content="Doc 2", metadata={"batch": "1"}),
]
ids = vector_store.add_documents(new_docs)
print(f"Added documents with IDs: {ids}")

# Add texts directly
texts = ["Text 1", "Text 2"]
metadatas = [{"source": "batch"}, {"source": "batch"}]
vector_store.add_texts(texts, metadatas=metadatas)

# Batch processing for large datasets
def batch_add_documents(vector_store, documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"Added batch {i//batch_size + 1}, documents {i+1} to {min(i+batch_size, len(documents))}")
```

### Document Preprocessing

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split large documents before adding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def preprocess_and_add(vector_store, documents):
    processed_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            processed_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            ))
    
    vector_store.add_documents(processed_docs)
    return processed_docs
```

## 4. Indexing and Optimization

### Index Management

```python
# Rebuild index (for supported stores)
def rebuild_index(vector_store):
    try:
        vector_store.rebuild_index()
        print("Index rebuilt successfully")
    except:
        print("Index rebuild not supported")

# Optimize index
def optimize_index(vector_store):
    try:
        vector_store.optimize()
        print("Index optimized")
    except:
        print("Index optimization not supported")

# Get index statistics
def get_index_stats(vector_store):
    try:
        stats = vector_store.index_stats()
        return stats
    except:
        # Manual stats for stores that don't support it
        sample_search = vector_store.similarity_search("test", k=1)
        return {
            "total_documents": len(sample_search) if sample_search else 0,
            "vector_store_type": type(vector_store).__name__
        }
```

### Performance Optimization

```python
# Configure search parameters for performance
class OptimizedVectorStore:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.search_cache = {}
    
    def cached_search(self, query, k=5, cache_ttl=300):
        import time
        import hashlib
        
        # Create cache key
        cache_key = hashlib.md5(f"{query}_{k}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.search_cache:
            result, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < cache_ttl:
                return result
        
        # Perform search
        result = self.vector_store.similarity_search(query, k=k)
        
        # Cache result
        self.search_cache[cache_key] = (result, time.time())
        
        return result
    
    def batch_search(self, queries, k=5):
        """Perform multiple searches efficiently"""
        results = []
        for query in queries:
            result = self.vector_store.similarity_search(query, k=k)
            results.append(result)
        return results
```

## 5. Metadata Operations

### Metadata Manipulation

```python
# Add metadata to existing documents
def add_metadata(vector_store, doc_filter, new_metadata):
    try:
        vector_store.update_metadata(doc_filter, new_metadata)
    except:
        print("Direct metadata update not supported")

# Search by metadata only
def search_by_metadata(vector_store, metadata_filter):
    try:
        return vector_store.get(filter=metadata_filter)
    except:
        # Fallback: search with dummy query and filter
        return vector_store.similarity_search("", k=1000, filter=metadata_filter)

# Get unique metadata values
def get_unique_metadata_values(vector_store, field):
    try:
        # For stores that support metadata queries
        return vector_store.get_unique_metadata(field)
    except:
        # Manual extraction
        results = vector_store.similarity_search("", k=1000)
        values = set()
        for doc in results:
            if field in doc.metadata:
                values.add(doc.metadata[field])
        return list(values)
```

### Metadata Indexing

```python
# Create metadata indexes for faster filtering
def create_metadata_index(vector_store, field):
    try:
        vector_store.create_index(field)
        print(f"Created index for field: {field}")
    except:
        print(f"Metadata indexing not supported for field: {field}")

# Metadata statistics
def get_metadata_stats(vector_store):
    try:
        all_docs = vector_store.get()
        metadata_stats = {}
        
        for doc in all_docs:
            for key, value in doc.metadata.items():
                if key not in metadata_stats:
                    metadata_stats[key] = {}
                if value not in metadata_stats[key]:
                    metadata_stats[key][value] = 0
                metadata_stats[key][value] += 1
        
        return metadata_stats
    except:
        return "Metadata statistics not available"
```

## 6. Persistence and Storage

### Save and Load Operations

```python
# Save vector store to disk
def save_vector_store(vector_store, path):
    try:
        vector_store.save_local(path)
        print(f"Vector store saved to {path}")
    except:
        print("Save operation not supported")

# Load vector store from disk
def load_vector_store(path, embeddings):
    try:
        vector_store = FAISS.load_local(path, embeddings)
        print(f"Vector store loaded from {path}")
        return vector_store
    except:
        print("Load operation not supported")
        return None

# Incremental saves
def incremental_save(vector_store, path, interval=100):
    import time
    save_counter = 0
    
    def save_checkpoint():
        nonlocal save_counter
        if save_counter % interval == 0:
            save_vector_store(vector_store, f"{path}_checkpoint_{save_counter}")
        save_counter += 1
    
    return save_checkpoint
```

### Backup and Recovery

```python
# Create backup
def create_backup(vector_store, backup_path):
    import shutil
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{backup_path}_backup_{timestamp}"
    
    try:
        vector_store.save_local(backup_name)
        print(f"Backup created: {backup_name}")
        return backup_name
    except Exception as e:
        print(f"Backup failed: {e}")
        return None

# Restore from backup
def restore_from_backup(backup_path, embeddings):
    try:
        vector_store = FAISS.load_local(backup_path, embeddings)
        print(f"Restored from backup: {backup_path}")
        return vector_store
    except Exception as e:
        print(f"Restore failed: {e}")
        return None
```

## 7. Advanced Operations

### Vector Store Merging

```python
# Merge multiple vector stores
def merge_vector_stores(main_store, *other_stores):
    """Merge multiple vector stores into one"""
    for store in other_stores:
        try:
            main_store.merge_from(store)
            print(f"Merged {type(store).__name__} into main store")
        except:
            # Manual merge for stores that don't support it
            try:
                docs = store.get()
                main_store.add_documents(docs)
                print(f"Manually merged {len(docs)} documents")
            except:
                print(f"Could not merge {type(store).__name__}")
    
    return main_store
```

### Similarity Analysis

```python
# Analyze document similarities
def analyze_document_similarities(vector_store, threshold=0.8):
    try:
        all_docs = vector_store.get()
        similarities = {}
        
        for i, doc1 in enumerate(all_docs):
            similar_docs = vector_store.similarity_search(
                doc1.page_content, k=5
            )
            
            similarities[i] = []
            for similar_doc in similar_docs[1:]:  # Skip self
                similarity_score = vector_store.similarity_search_with_score(
                    similar_doc.page_content, k=1
                )[0][1]
                
                if similarity_score >= threshold:
                    similarities[i].append({
                        'document': similar_doc,
                        'similarity': similarity_score
                    })
        
        return similarities
    except:
        return "Similarity analysis not available"

# Find duplicate documents
def find_duplicates(vector_store, threshold=0.95):
    similarities = analyze_document_similarities(vector_store, threshold)
    duplicates = []
    
    for doc_id, similar_docs in similarities.items():
        if similar_docs:
            duplicates.append({
                'document_id': doc_id,
                'duplicates': similar_docs
            })
    
    return duplicates
```

### Clustering Operations

```python
# Cluster documents
def cluster_documents(vector_store, n_clusters=5):
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Get all documents and their vectors
        all_docs = vector_store.get()
        vectors = []
        
        for doc in all_docs:
            vector = vector_store.embeddings.embed_query(doc.page_content)
            vectors.append(vector)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)
        
        # Group documents by cluster
        clustered_docs = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_docs:
                clustered_docs[cluster_id] = []
            clustered_docs[cluster_id].append(all_docs[i])
        
        return clustered_docs
    except ImportError:
        print("sklearn required for clustering")
        return None
```

## 8. Vector Store Specific Methods

### FAISS Specific Operations

```python
# FAISS specific methods
def faiss_operations(faiss_store):
    # Get index
    index = faiss_store.index
    
    # Index statistics
    print(f"Total vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")
    
    # Add vectors directly
    vectors = np.random.random((10, index.d)).astype('float32')
    index.add(vectors)
    
    # Search with custom parameters
    scores, indices = index.search(vectors[:1], k=5)
    
    # Save/load index
    import faiss
    faiss.write_index(index, "faiss_index.bin")
    loaded_index = faiss.read_index("faiss_index.bin")
```

### Chroma Specific Operations

```python
# Chroma specific methods
def chroma_operations(chroma_store):
    # Get collection
    collection = chroma_store._collection
    
    # Collection statistics
    print(f"Collection name: {collection.name}")
    print(f"Document count: {collection.count()}")
    
    # Peek at collection
    peek_result = collection.peek()
    print(f"Sample data: {peek_result}")
    
    # Query with where clause
    results = collection.query(
        query_texts=["machine learning"],
        n_results=5,
        where={"type": "tech"}
    )
    
    # Get all documents
    all_docs = collection.get()
    
    # Delete collection
    # chroma_store._client.delete_collection(collection.name)
```

### Pinecone Specific Operations

```python
# Pinecone specific operations (if using Pinecone)
def pinecone_operations(pinecone_store):
    try:
        # Get index stats
        stats = pinecone_store.index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        # Upsert vectors
        vectors = [
            ("id1", [0.1, 0.2, 0.3], {"type": "test"}),
            ("id2", [0.4, 0.5, 0.6], {"type": "test"})
        ]
        pinecone_store.index.upsert(vectors)
        
        # Query with metadata filter
        results = pinecone_store.index.query(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            filter={"type": "test"}
        )
        
        # Delete by ID
        pinecone_store.index.delete(ids=["id1", "id2"])
        
    except Exception as e:
        print(f"Pinecone operations failed: {e}")
```

## Usage Examples

```python
# Complete example demonstrating multiple operations
def demonstrate_vector_store_operations():
    # Initialize
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    documents = [
        Document(page_content="Python programming", metadata={"type": "tech", "difficulty": "beginner"}),
        Document(page_content="Machine learning algorithms", metadata={"type": "tech", "difficulty": "advanced"}),
        Document(page_content="Beautiful sunset", metadata={"type": "nature", "difficulty": "easy"})
    ]
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Demonstrate operations
    print("=== Basic Search ===")
    results = vector_store.similarity_search("programming", k=2)
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    print("\n=== Search with Scores ===")
    results = vector_store.similarity_search_with_score("programming", k=2)
    for doc, score in results:
        print(f"Score: {score:.3f}, Content: {doc.page_content}")
    
    print("\n=== Filtered Search ===")
    tech_results = vector_store.similarity_search(
        "algorithms", k=2, filter={"type": "tech"}
    )
    for doc in tech_results:
        print(f"Tech doc: {doc.page_content}")
    
    print("\n=== MMR Search ===")
    mmr_results = vector_store.max_marginal_relevance_search(
        "programming", k=2, fetch_k=5
    )
    for doc in mmr_results:
        print(f"MMR result: {doc.page_content}")
    
    print("\n=== Adding New Documents ===")
    new_docs = [
        Document(page_content="JavaScript frameworks", metadata={"type": "tech", "difficulty": "intermediate"})
    ]
    vector_store.add_documents(new_docs)
    
    print("\n=== Updated Search Results ===")
    updated_results = vector_store.similarity_search("programming", k=3)
    for doc in updated_results:
        print(f"Updated: {doc.page_content}")

# Run the demonstration
if __name__ == "__main__":
    demonstrate_vector_store_operations()
```

This comprehensive guide covers all major vector store operations in LangChain. Each method includes practical examples and error handling for different vector store implementations. The operations are organized by functionality, making it easy to find specific actions you need to perform on your vector stores.