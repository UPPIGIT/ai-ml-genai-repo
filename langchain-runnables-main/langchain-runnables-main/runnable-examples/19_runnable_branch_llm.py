"""
Example 19: RunnableBranch + LLM Integration
===========================================

This example demonstrates using RunnableBranch to route input to different LLM prompts based on user intent (question, greeting, or command).
Requires OpenAI API key set in your environment.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

# Intent classifier (very basic demo)
def classify_intent(text):
    t = text.lower()
    if t.startswith("hi") or t.startswith("hello"):
        return "greeting"
    if t.endswith("?"):
        return "question"
    return "command"

# LLM chains for each intent
def get_greeting_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    prompt = ChatPromptTemplate.from_template("Respond to the greeting: {input}")
    return prompt | llm | StrOutputParser()

def get_question_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("Answer the question: {input}")
    return prompt | llm | StrOutputParser()

def get_command_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Acknowledge and confirm the command: {input}")
    return prompt | llm | StrOutputParser()

if __name__ == "__main__":
    classifier = RunnableLambda(classify_intent)
    branch = RunnableBranch(
        (lambda x: classifier.invoke(x) == "greeting", get_greeting_chain()),
        (lambda x: classifier.invoke(x) == "question", get_question_chain()),
        (lambda x: True, get_command_chain()),
    )
    samples = [
        "Hello there!",
        "What is the capital of France?",
        "Turn off the lights",
    ]
    print("\n=== RunnableBranch + LLM Example ===")
    for s in samples:
        print(f"Input: {s}")
        print("Output:", branch.invoke({"input": s}))
        print("-") 