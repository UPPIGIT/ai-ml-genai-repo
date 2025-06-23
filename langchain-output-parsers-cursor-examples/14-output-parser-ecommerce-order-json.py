"""
Example 14: E-commerce Order Parser (with Gemini)
Parses an order confirmation email into a structured JSON object.
"""
from langchain.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt = PromptTemplate.from_template(
    """Extract the following fields from this order confirmation email as JSON:
    - order_id
    - customer_name
    - items (list of {name, quantity, price})
    - total
    Email: Hi John, your order #12345 for 2x Widget ($10 each) and 1x Gadget ($20) has shipped. Total: $40."""
)

parser = JsonOutputParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed order:", result) 