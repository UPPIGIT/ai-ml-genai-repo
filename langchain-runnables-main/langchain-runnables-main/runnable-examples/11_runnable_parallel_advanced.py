"""
Example 11: Advanced RunnableParallel - Text Feature Extraction
=============================================================

This example demonstrates how RunnableParallel can be used to extract multiple features from text in parallel, such as sentiment, word count, and language detection.
"""

from langchain_core.runnables import RunnableLambda, RunnableParallel

# Sentiment analysis (very basic demo)
def sentiment(text):
    pos = ["good", "happy", "love", "excellent", "great"]
    neg = ["bad", "sad", "hate", "terrible", "poor"]
    t = text.lower()
    if any(w in t for w in pos):
        return "positive"
    if any(w in t for w in neg):
        return "negative"
    return "neutral"

# Word count
def word_count(text):
    return len(text.split())

# Language detection (very naive)
def detect_language(text):
    if any(ord(c) > 128 for c in text):
        return "non-english"
    return "english"

if __name__ == "__main__":
    parallel = RunnableParallel({
        "sentiment": RunnableLambda(sentiment),
        "word_count": RunnableLambda(word_count),
        "language": RunnableLambda(detect_language),
    })
    sample = "I love LangChain! This is excellent."
    print("\n=== Advanced RunnableParallel: Text Feature Extraction ===")
    print(f"Input: {sample}")
    result = parallel.invoke(sample)
    print("Output:", result) 