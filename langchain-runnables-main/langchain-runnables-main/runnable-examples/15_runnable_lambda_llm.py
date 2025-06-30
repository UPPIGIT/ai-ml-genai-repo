"""
Example 15: RunnableLambda + LLM Integration
===========================================

This example demonstrates using RunnableLambda to post-process LLM output.
Requires OpenAI API key set in your environment.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import re

# Step 1: LLM chain to generate a product description
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
prompt = ChatPromptTemplate.from_template(
    "Write a short product description for: {product}."
)
output_parser = StrOutputParser()
llm_chain = prompt | llm | output_parser

# Step 2: RunnableLambda to extract keywords from the LLM output
def extract_keywords(text):
    # Simple keyword extraction: pick capitalized words and nouns (demo only)
    return re.findall(r"\\b[A-Z][a-z]+\\b", text)
extractor = RunnableLambda(extract_keywords)

def example_llm_lambda():
    print("\n=== RunnableLambda + LLM Example ===")
    product = "wireless noise-cancelling headphones"
    description = llm_chain.invoke({"product": product})
    print("LLM Output:", description)
    keywords = extractor.invoke(description)
    print("Extracted Keywords:", keywords)

if __name__ == "__main__":
    example_llm_lambda() 