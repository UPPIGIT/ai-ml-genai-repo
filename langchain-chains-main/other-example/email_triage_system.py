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

class DeptOut(BaseModel):
    department: Literal['technical', 'hr', 'finance'] = Field(..., description='Department classification')

dept_parser = PydanticOutputParser(pydantic_object=DeptOut)

prompt_classify = PromptTemplate(
    template='Classify this internal company email as technical, hr, or finance:\n{email}\n{format_instruction}',
    input_variables=['email'],
    partial_variables={'format_instruction': dept_parser.get_format_instructions()}
)

# Department responses
prompt_tech = PromptTemplate(
    template='Reply: We'll loop in engineering.',
    input_variables=[]
)
prompt_hr = PromptTemplate(
    template='Reply: HR will get back to you shortly.',
    input_variables=[]
)
prompt_fin = PromptTemplate(
    template='Reply: We've passed this to the finance team.',
    input_variables=[]
)

branch = RunnableBranch(
    (lambda x: x.department == 'technical', prompt_tech | model | parser),
    (lambda x: x.department == 'hr', prompt_hr | model | parser),
    (lambda x: x.department == 'finance', prompt_fin | model | parser),
    RunnableLambda(lambda x: 'Could not classify this email.')
)

chain = prompt_classify | model | dept_parser | branch

email = "Hi, I need help with my payroll for last month. There seems to be a discrepancy."

print(chain.invoke({'email': email})) 