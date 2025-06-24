from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class SentimentOut(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(..., description='Sentiment of the review')

sentiment_parser = PydanticOutputParser(pydantic_object=SentimentOut)

# Sentiment classification
prompt_sentiment = PromptTemplate(
    template='Classify the sentiment (positive or negative) of this product review:\n{review}\n{format_instruction}',
    input_variables=['review'],
    partial_variables={'format_instruction': sentiment_parser.get_format_instructions()}
)

# Keyword extraction
prompt_keywords = PromptTemplate(
    template='Extract 3-5 important keywords from this product review:\n{review}',
    input_variables=['review']
)

parallel = RunnableParallel({
    'sentiment': prompt_sentiment | model | sentiment_parser,
    'keywords': prompt_keywords | model | parser
})

# Response templates
prompt_thank = PromptTemplate(
    template='Write a thank-you note for this positive review. Mention these keywords: {keywords}',
    input_variables=['keywords']
)
prompt_apology = PromptTemplate(
    template='Write an apology and ask for more info about these issues: {keywords}',
    input_variables=['keywords']
)

branch = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'positive', prompt_thank | model | parser),
    (lambda x: x['sentiment'].sentiment == 'negative', prompt_apology | model | parser),
    RunnableLambda(lambda x: 'Could not determine sentiment.')
)

chain = parallel | branch

review = "The headphones sound great and are very comfortable, but the Bluetooth connection keeps dropping."

print(chain.invoke({'review': review})) 