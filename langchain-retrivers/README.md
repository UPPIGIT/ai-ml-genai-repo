# LangChain Retriever Examples (Open Source)

This repository contains step-by-step examples of the most popular and relevant retrievers in [LangChain](https://python.langchain.com/), using only open-source and HuggingFace models. Each example is numbered and progresses from simple to advanced, with clear code comments and explanations.

## Examples Included

| File                                 | Description                                                        |
|--------------------------------------|--------------------------------------------------------------------|
| **01_vectorstore_retriever.py**      | Basic Vector Store Retriever using Chroma and HuggingFace embeddings. |
| **02_wikipedia_retriever.py**        | Retrieve information from Wikipedia using the open-source retriever. |
| **03_mmr_retriever.py**              | Maximal Marginal Relevance (MMR) Retriever for diverse results.      |
| **04_multiquery_retriever.py**       | MultiQueryRetriever with HuggingFace LLM and embeddings.             |
| **05_contextual_compression_retriever.py** | ContextualCompressionRetriever with HuggingFace LLM and embeddings. |
| **06_bm25_retriever.py**             | BM25Retriever for classic keyword-based retrieval.                   |
| **07_self_query_retriever.py**       | SelfQueryRetriever for natural language filtering on metadata.       |
| **08_ensemble_retriever.py**         | EnsembleRetriever combining BM25 and VectorStore retrievers.         |
| **09_time_weighted_retriever.py**    | TimeWeightedVectorStoreRetriever for recency-aware retrieval.        |

## Retriever Types Covered
- Vector Store Retriever (Chroma/FAISS)
- WikipediaRetriever
- Maximal Marginal Relevance (MMR) Retriever
- MultiQueryRetriever
- ContextualCompressionRetriever
- BM25Retriever
- SelfQueryRetriever
- EnsembleRetriever
- TimeWeightedVectorStoreRetriever

## Setup

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This will install LangChain, HuggingFace, FAISS, Chroma, Wikipedia, and all required libraries for open-source usage.

## Usage

Run any example directly with Python. For example:
```bash
python 01_vectorstore_retriever.py
```

Each script is self-contained and prints results to the console. You can modify the queries or documents to experiment further.

## Notes
- All embeddings and LLMs use open-source HuggingFace models (e.g., `sentence-transformers/all-mpnet-base-v2`, `google/flan-tiny`).
- No API keys or proprietary services are required.
- WikipediaRetriever fetches live data from Wikipedia.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## License
This project is open source and free to use for educational and research purposes. 