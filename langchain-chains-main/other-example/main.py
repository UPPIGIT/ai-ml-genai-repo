import os
from langchain_openai import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate

# Ensure you have set your OpenAI API key as an environment variable:
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"


def basic_llm_chain():
    """Basic LLMChain: Single prompt, single output."""
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="What is {topic}?"
    )
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"topic": "LangChain"})
    print("\n--- Basic LLMChain Example ---")
    print(result["text"].strip())


def simple_sequential_chain():
    """SimpleSequentialChain: Output of one chain is input to the next."""
    llm = OpenAI(temperature=0)
    chain1 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["topic"],
            template="What is {topic}?"
        )
    )
    chain2 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="Summarize this in one sentence: {text}"
        )
    )
    overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
    print("\n--- SimpleSequentialChain Example ---")
    result = overall_chain.invoke("LangChain")
    print(result)


def multi_input_output_chain():
    """SequentialChain: Multiple inputs and outputs, more advanced chaining."""
    llm = OpenAI(temperature=0)
    # First chain: Generate a company name
    chain1 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["product"],
            template="What would be a good company name for a company that makes {product}?"
        ),
        output_key="company_name"
    )
    # Second chain: Write a tagline for the company
    chain2 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["company_name"],
            template="Write a tagline for the company {company_name}."
        ),
        output_key="tagline"
    )
    # Third chain: Write a short description using both
    chain3 = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["company_name", "tagline"],
            template="Write a short description for {company_name} with the tagline: {tagline}"
        ),
        output_key="description"
    )
    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3],
        input_variables=["product"],
        output_variables=["company_name", "tagline", "description"],
        verbose=True
    )
    print("\n--- SequentialChain (Multi-input/output) Example ---")
    result = overall_chain.invoke({"product": "AI-powered toothbrush"})
    print(result)


def main():
    basic_llm_chain()
    simple_sequential_chain()
    multi_input_output_chain()


if __name__ == "__main__":
    main()
