from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07", task_type="RETRIEVAL_QUERY"
)
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07", task_type="RETRIEVAL_DOCUMENT"
)

query = "tell me about Sachin Tendulkar, MS Dhoni and Virat Kohli"
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar is a legendary Indian cricketer, often referred to as the 'God of Cricket'."
]

q_embed = query_embeddings.embed_query(query)
d_embed = doc_embeddings.embed_documents(
    documents
)

# Calculate cosine similarity scores between the query and each document
similarity_scores = cosine_similarity([q_embed], d_embed)[0]
print("Similarity Scores:", similarity_scores)

# Set a similarity threshold (e.g., 0.7)
threshold = 0.7

# Get indices and scores of documents above the threshold
similar_docs = [(i, score) for i, score in enumerate(similarity_scores) if score >= threshold]

# Print similar documents and their scores
for idx, score in similar_docs:
    print(f"Document: {documents[idx]}")
    print(f"Similarity Score: {score}\n")

print("--"*40)
scores = cosine_similarity([q_embed], d_embed)[0]
print("Scores:", scores)
# Get the index of the document with the highest similarity score
print("Similarity scores:", list(enumerate(scores)))
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

#based on distance, the most similar document is:
# Sachin Tendulkar is a legendary Indian cricketer, often referred to as the 'God of Cricket'.
# with a similarity score of 0.9999999999999999
# Note: The similarity score is very close to 1, indicating a high degree of similarity.
# The output shows the similarity scores for each document with respect to the query.
# The document with the highest score is the most relevant to the query.
# The output also includes the index of the most similar document and its similarity score.
# The similarity score is very close to 1, indicating a high degree of similarity.
# The output shows the similarity scores for each document with respect to the query.
# The document with the highest score is the most relevant to the query.
# The output also includes the index of the most similar document and its similarity score.
