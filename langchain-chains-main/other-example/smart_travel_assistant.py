from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class IntentOut(BaseModel):
    intent: Literal['book_flight', 'find_hotel', 'cancel_trip'] = Field(..., description='User intent')

intent_parser = PydanticOutputParser(pydantic_object=IntentOut)

prompt_intent = PromptTemplate(
    template='Extract the user intent (book_flight, find_hotel, cancel_trip) from this message:\n{message}\n{format_instruction}',
    input_variables=['message'],
    partial_variables={'format_instruction': intent_parser.get_format_instructions()}
)

# Task-specific chains
prompt_flight = PromptTemplate(
    template='Book a flight for the user and provide a confirmation message.',
    input_variables=[]
)
prompt_hotel = PromptTemplate(
    template='Find and recommend a hotel for the user near their specified location.',
    input_variables=[]
)
prompt_cancel = PromptTemplate(
    template='Confirm the cancellation of the user\'s trip and provide a polite message.',
    input_variables=[]
)

branch = RunnableBranch(
    (lambda x: x.intent == 'book_flight', prompt_flight | model | parser),
    (lambda x: x.intent == 'find_hotel', prompt_hotel | model | parser),
    (lambda x: x.intent == 'cancel_trip', prompt_cancel | model | parser),
    RunnableLambda(lambda x: 'Sorry, I could not understand your request.')
)

chain = prompt_intent | model | intent_parser | branch

message = "Book me a flight to Paris"

print(chain.invoke({'message': message})) 