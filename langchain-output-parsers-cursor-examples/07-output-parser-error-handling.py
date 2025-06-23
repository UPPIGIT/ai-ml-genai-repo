"""
Example 7: Robust Output Parser with Error Handling (with Gemini)
This example shows how to handle parsing errors and fallback to a default value.
"""
from langchain.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
prompt = PromptTemplate.from_template(
    "Provide a JSON object with keys: name, age, city. If you can't, just say 'N/A'."
)
parser = JsonOutputParser()

chain = prompt | model

try:
    output = chain.invoke({})
    result = parser.parse(output)
except Exception as e:
    print("Parsing failed, using fallback. Error:", e)
    result = {"name": None, "age": None, "city": None}

print("Parsed (with error handling):", result) 