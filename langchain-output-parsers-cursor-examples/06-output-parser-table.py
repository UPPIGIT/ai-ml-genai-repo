"""
Example 6: Table Output Parser (CSV to List of Dicts, with Gemini)
This example demonstrates how to parse a table (CSV) output from Gemini into a list of dictionaries.
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
    "List three countries and their capitals in CSV format with columns Country,Capital."
)

class TableCSVParser(StrOutputParser):
    def parse(self, text: str):
        reader = csv.DictReader(io.StringIO(text.strip()))
        return [row for row in reader]

parser = TableCSVParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed table:", result) 