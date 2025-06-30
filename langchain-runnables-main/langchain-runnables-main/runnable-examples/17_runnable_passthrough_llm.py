"""
Example 17: RunnablePassthrough + LLM Integration
===============================================

This example demonstrates using RunnablePassthrough to log LLM input/output and to enrich LLM input with metadata.
Requires OpenAI API key set in your environment.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Logger function
def log_input(x):
    print(f"[LOG] LLM Input: {x}")
    return x

def log_output(x):
    print(f"[LOG] LLM Output: {x}")
    return x

# Enrich input with metadata
def enrich_input(data):
    # Add a 'source' field
    data = dict(data)
    data['source'] = 'user_query'
    return data

def get_llm_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question. Source: {source}. Question: {question}"
    )
    return prompt | llm | StrOutputParser()

if __name__ == "__main__":
    passthrough = RunnablePassthrough()
    logger_in = RunnableLambda(log_input)
    logger_out = RunnableLambda(log_output)
    enricher = RunnableLambda(enrich_input)
    llm_chain = get_llm_chain()
    # Chain: enrich input -> log input -> LLM -> log output
    chain = enricher | logger_in | llm_chain | logger_out
    sample = {"question": "What is LangChain?"}
    print("\n=== RunnablePassthrough + LLM Example ===")
    result = chain.invoke(sample)
    print("Final Output:", result) 