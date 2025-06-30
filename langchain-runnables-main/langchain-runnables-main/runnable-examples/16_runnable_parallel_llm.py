"""
Example 16: RunnableParallel + LLM Integration
=============================================

This example demonstrates using RunnableParallel to run an LLM summary, sentiment analysis, and word count in parallel.
Requires OpenAI API key set in your environment.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# LLM summary chain
def get_summary_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )
    return prompt | llm | StrOutputParser()

# Sentiment analysis (simple demo)
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

if __name__ == "__main__":
    parallel = RunnableParallel({
        "summary": get_summary_chain(),
        "sentiment": RunnableLambda(sentiment),
        "word_count": RunnableLambda(word_count),
    })
    sample = "LangChain is a powerful Python library for building applications with large language models. I love using it!"
    print("\n=== RunnableParallel + LLM Example ===")
    print(f"Input: {sample}")
    result = parallel.invoke(sample)
    print("Output:", result) 