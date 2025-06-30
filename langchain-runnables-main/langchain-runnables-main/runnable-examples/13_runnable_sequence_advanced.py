"""
Example 13: Advanced RunnableSequence - Multi-step Data Enrichment
=================================================================

This example demonstrates how RunnableSequence can be used to build a multi-step data enrichment pipeline: clean text, extract entities, and summarize.
"""

import re
from langchain_core.runnables import RunnableLambda, RunnableSequence

# Step 1: Clean text (lowercase, remove punctuation)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()

# Step 2: Extract entities (very basic: capitalized words in original text)
def extract_entities(text):
    # For demo, just return words that start with uppercase in the original text
    return [w for w in text.split() if w.istitle()]

# Step 3: Summarize (very basic: return first 5 words)
def summarize(text):
    words = text.split()
    return " ".join(words[:5]) + ("..." if len(words) > 5 else "")

if __name__ == "__main__":
    sequence = RunnableSequence([
        RunnableLambda(clean_text),
        RunnableLambda(summarize),
    ])
    sample = "LangChain is a Python library for building applications with LLMs. It is developed by Harrison Chase."
    print("\n=== Advanced RunnableSequence: Data Enrichment Pipeline ===")
    print(f"Input: {sample}")
    result = sequence.invoke(sample)
    print("Output (cleaned & summarized):", result)

    # Entity extraction as a separate step
    print("\nEntity Extraction:")
    entities = RunnableLambda(extract_entities).invoke(sample)
    print("Entities:", entities) 