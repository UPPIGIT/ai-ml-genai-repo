from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Step 1: Translate English text to French
template_translate = PromptTemplate(
    template='Translate the following text to French:\n{text}',
    input_variables=['text']
)

# Step 2: Summarize the French translation
template_summarize = PromptTemplate(
    template='Summarize the following French text in 2 sentences:\n{text}',
    input_variables=['text']
)

model = ChatOpenAI()
parser = StrOutputParser()

# Chain: Translate -> Summarize
chain = template_translate | model | parser | template_summarize | model | parser

input_text = "Artificial intelligence is transforming the world by enabling machines to learn from data and make decisions."

result = chain.invoke({'text': input_text})

print(result)

chain.get_graph().print_ascii() 