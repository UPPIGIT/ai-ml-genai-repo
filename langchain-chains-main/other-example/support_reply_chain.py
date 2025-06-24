from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Branch 1: Extract issue summary
prompt_summary = PromptTemplate(
    template='Summarize the main issue described in this customer support email:\n{email}',
    input_variables=['email']
)

# Branch 2: Detect customer sentiment
prompt_sentiment = PromptTemplate(
    template='Detect the sentiment (positive, negative, or neutral) of this customer support email:\n{email}',
    input_variables=['email']
)

# Parallel chain for summary and sentiment
dual_chain = RunnableParallel({
    'issue_summary': prompt_summary | model | parser,
    'sentiment': prompt_sentiment | model | parser
})

# Combine and generate agent reply
prompt_reply = PromptTemplate(
    template='You are a customer support agent. Write a personalized reply based on the following:\nIssue summary: {issue_summary}\nCustomer sentiment: {sentiment}',
    input_variables=['issue_summary', 'sentiment']
)

reply_chain = prompt_reply | model | parser

# Full chain: parallel extraction -> reply generation
chain = dual_chain | reply_chain

email = '''\
Hello,

I'm really frustrated. My internet has been down for two days and I work from home. I've tried restarting my router several times but nothing works. Please help me resolve this as soon as possible.

Thanks,
Alex
'''

result = chain.invoke({'email': email})

print(result)

chain.get_graph().print_ascii() 