"""
Example 15: Financial Report Table Parser (with Gemini)
Parses a CSV table of quarterly revenue into a list of dictionaries.
"""
from langchain.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import csv
import io

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt = PromptTemplate.from_template(
    "Provide a CSV table of quarterly revenue for 2023 with columns Quarter,Revenue."
)

class FinancialTableParser(StrOutputParser):
    def parse(self, text: str):
        reader = csv.DictReader(io.StringIO(text.strip()))
        return [row for row in reader]

parser = FinancialTableParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed financial table:", result) 