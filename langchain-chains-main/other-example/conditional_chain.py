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

class TicketClassification(BaseModel):
    priority: Literal['urgent', 'normal'] = Field(description='Priority of the support ticket')

parser2 = PydanticOutputParser(pydantic_object=TicketClassification)

prompt1 = PromptTemplate(
    template='Classify the following support ticket as urgent or normal:\n{ticket}\n{format_instruction}',
    input_variables=['ticket'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write a quick response for an URGENT support ticket:\n{ticket}',
    input_variables=['ticket']
)

prompt3 = PromptTemplate(
    template='Write a polite response for a NORMAL support ticket:\n{ticket}',
    input_variables=['ticket']
)

branch_chain = RunnableBranch(
    (lambda x: x.priority == 'urgent', prompt2 | model | parser),
    (lambda x: x.priority == 'normal', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not classify ticket priority.")
)

chain = classifier_chain | branch_chain

test_ticket = "My server is down and I can't access my website!"

print(chain.invoke({'ticket': test_ticket}))

chain.get_graph().print_ascii()