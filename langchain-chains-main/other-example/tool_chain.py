from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Step 1: Extract math expression from user query
prompt_extract = PromptTemplate(
    template='Extract the mathematical expression from the following query:\n{query}',
    input_variables=['query']
)

# Step 2: Evaluate the expression using Python
def evaluate_math(inputs):
    try:
        result = eval(inputs['expression'], {"__builtins__": {}})
        return {'result': str(result)}
    except Exception as e:
        return {'result': f'Error: {e}'}

eval_tool = RunnableLambda(lambda x: evaluate_math({'expression': x}))

# Step 3: LLM explains the result
prompt_explain = PromptTemplate(
    template='Explain the result of this calculation: {expression} = {result}',
    input_variables=['expression', 'result']
)

# Chain: Extract expression -> Evaluate -> Explain
chain = (
    prompt_extract | model | parser |
    eval_tool |
    (lambda x: {'expression': x, 'result': eval_tool.invoke(x) if isinstance(x, str) else x['result']}) |
    prompt_explain | model | parser
)

user_query = "What is the result of 15 * (3 + 2)?"

# Extract expression
expression = (prompt_extract | model | parser).invoke({'query': user_query})
# Evaluate
result = evaluate_math({'expression': expression})
# Explain
final = (prompt_explain | model | parser).invoke({'expression': expression, 'result': result['result']})

print(final)

# For graph visualization, show the main steps
# (Graph for the full chain with lambda/eval is not supported, so we show the main LLM steps)
print("\nChain steps:")
print("1. Extract expression from query\n2. Evaluate with Python\n3. LLM explains the result") 