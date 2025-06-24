from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Step 1: Extract relevant context from the document
template_context = PromptTemplate(
    template='Given the following document:\n{document}\n\nAnd the question: {question}\n\nExtract the most relevant part of the document to answer the question.',
    input_variables=['document', 'question']
)

# Step 2: Answer the question using the extracted context
template_answer = PromptTemplate(
    template='Using this context:\n{context}\n\nAnswer the question: {question}',
    input_variables=['context', 'question']
)

model = ChatOpenAI()
parser = StrOutputParser()

# Chain: Extract context -> Answer question
chain = template_context | model | parser | template_answer | model | parser

document = '''
The Amazon rainforest is the largest tropical rainforest in the world, covering over five and a half million square kilometers. It is home to an unparalleled diversity of plant and animal species. The rainforest plays a critical role in regulating the Earth's climate and is often referred to as the "lungs of the planet" because it produces about 20% of the world's oxygen. Deforestation and climate change are major threats to the Amazon's future.
'''

question = "Why is the Amazon rainforest important for the Earth's climate?"

result = chain.invoke({'document': document, 'question': question})

print(result)

chain.get_graph().print_ascii() 