"""
Example 10: Advanced RunnableLambda - Text Cleaning Pipeline
===========================================================

This example demonstrates how RunnableLambda can be used to build a text cleaning and normalization pipeline for NLP tasks.
"""

import re
from langchain_core.runnables import RunnableLambda

# Example stopwords list (for demo purposes)
STOPWORDS = {"the", "is", "at", "which", "on", "and", "a", "an", "of", "to", "in"}

# Step 1: Lowercase the text
lowercase = RunnableLambda(lambda s: s.lower())

# Step 2: Remove punctuation
remove_punct = RunnableLambda(lambda s: re.sub(r"[\W_]+", " ", s))

# Step 3: Remove stopwords
def remove_stopwords(s):
    words = s.split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered)
remove_stop = RunnableLambda(remove_stopwords)

# Step 4: Strip extra spaces
strip_spaces = RunnableLambda(lambda s: " ".join(s.split()))

def text_cleaning_pipeline(text):
    """Runs the text through the cleaning pipeline."""
    print("Original:", text)
    s = lowercase.invoke(text)
    print("Lowercased:", s)
    s = remove_punct.invoke(s)
    print("No punctuation:", s)
    s = remove_stop.invoke(s)
    print("No stopwords:", s)
    s = strip_spaces.invoke(s)
    print("Stripped spaces:", s)
    return s

if __name__ == "__main__":
    sample = "The quick brown fox, at the zoo, jumps on a lazy dog!"
    print("\n=== Advanced RunnableLambda: Text Cleaning Pipeline ===")
    cleaned = text_cleaning_pipeline(sample)
    print("\nFinal cleaned text:", cleaned) 