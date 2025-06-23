"""
Example 16: Healthcare Patient Summary Parser (with Gemini)
Parses a doctor's note into a structured patient summary.
"""
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

class PatientSummary(BaseModel):
    name: str = Field(description="Patient's name")
    age: int = Field(description="Patient's age")
    diagnosis: str = Field(description="Diagnosis")
    medications: list[str] = Field(description="List of medications")

prompt = PromptTemplate.from_template(
    "Summarize the following doctor's note as JSON with fields: name, age, diagnosis, medications. Note: Jane Doe, 45, diagnosed with hypertension. Prescribed Lisinopril."
)

parser = StructuredOutputParser.from_schema(PatientSummary)
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed patient summary:", result) 