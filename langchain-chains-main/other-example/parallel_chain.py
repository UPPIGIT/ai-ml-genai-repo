from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Branch 1: Summarize the article
prompt_summary = PromptTemplate(
    template='Summarize the following article in 2-3 sentences:\n{article}',
    input_variables=['article']
)

# Branch 2: Extract top 5 keywords
prompt_keywords = PromptTemplate(
    template='Extract the top 5 keywords from the following article:\n{article}',
    input_variables=['article']
)

# Branch 3: Detect sentiment
prompt_sentiment = PromptTemplate(
    template='Detect the sentiment (positive, negative, or neutral) of the following article:\n{article}',
    input_variables=['article']
)

parallel_chain = RunnableParallel({
    'summary': prompt_summary | model | parser,
    'keywords': prompt_keywords | model | parser,
    'sentiment': prompt_sentiment | model | parser
})

article = '''\
The city council has approved a new plan to expand green spaces throughout the city. The initiative aims to plant thousands of trees, create new parks, and improve air quality. Residents have expressed strong support, citing the benefits for health and community well-being. However, some business owners are concerned about potential disruptions during construction. The project is expected to begin next spring and will take several years to complete.
'''

result = parallel_chain.invoke({'article': article})

print(result)

parallel_chain.get_graph().print_ascii()

