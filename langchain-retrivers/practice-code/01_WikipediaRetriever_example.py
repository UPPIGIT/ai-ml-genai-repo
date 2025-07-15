from langchain_community.retrievers import WikipediaRetriever

# Initialize the Wikipedia Retriever
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Example query to retrieve information from Wikipedia
query = "sachin tendulkar"
documents = retriever.invoke(query)

for i ,doc in enumerate(documents):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters of content
    print("\n")
    