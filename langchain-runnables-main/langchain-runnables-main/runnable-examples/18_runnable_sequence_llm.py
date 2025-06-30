"""
Example 18: RunnableSequence + LLM Integration
=============================================

This example demonstrates using RunnableSequence to build a multi-step LLM pipeline: prompt -> LLM -> post-process -> LLM follow-up.
Requires OpenAI API key set in your environment.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

# Step 1: Initial prompt and LLM response
def get_initial_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
    prompt = ChatPromptTemplate.from_template(
        "Explain the main idea of: {topic}"
    )
    return prompt | llm | StrOutputParser()

# Step 2: Post-process (extract first sentence)
def extract_first_sentence(text):
    return text.split(".")[0] + "."

# Step 3: LLM follow-up (ask for an analogy)
def get_analogy_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
    prompt = ChatPromptTemplate.from_template(
        "Give an analogy for this idea: {idea}"
    )
    return prompt | llm | StrOutputParser()

if __name__ == "__main__":
    sequence = RunnableSequence([
        get_initial_chain(),
        RunnableLambda(extract_first_sentence),
        get_analogy_chain(),
    ])
    sample = {"topic": "blockchain technology"}
    print("\n=== RunnableSequence + LLM Example ===")
    result = sequence.invoke(sample)
    print("Output (analogy):", result) 