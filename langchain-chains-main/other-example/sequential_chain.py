from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a unique business idea in the field of {industry}',
    input_variables=['industry']
)

prompt2 = PromptTemplate(
    template='Summarize the following business idea in 3 bullet points:\n{text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'industry': 'sustainable energy'})

print(result)

chain.get_graph().print_ascii()