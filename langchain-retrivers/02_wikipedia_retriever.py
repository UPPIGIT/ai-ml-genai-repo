"""
02_wikipedia_retriever.py
Example: Using WikipediaRetriever in LangChain
Step-by-step with comments. Uses only open-source components.
"""

# 1. Import WikipediaRetriever from the community package
from langchain_community.retrievers import WikipediaRetriever

# 2. Create the retriever (no API key needed for basic usage)
wikipedia_retriever = WikipediaRetriever(top_k_results=2, lang="en")

# 3. Define a query
query = "Python programming language"

# 4. Retrieve relevant documents from Wikipedia
results = wikipedia_retriever.invoke(query)

# 5. Print the results
print("Query:", query)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content[:200]}...")  # Print first 200 chars 