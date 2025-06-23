"""
Example 13: Customer Support Chat Summary Parser (with Gemini)
Extracts structured summary, sentiment, and action items from a customer support chat.
"""
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt = PromptTemplate.from_template(
    """Summarize the following customer support chat. Extract:
    - summary (1-2 sentences)
    - sentiment (positive/negative/neutral)
    - action items (as a list)
    Chat: Customer: My order hasn't arrived. Agent: Sorry! We'll resend it today."""
)

class SupportSummaryParser(BaseOutputParser):
    def parse(self, text: str):
        summary = re.search(r"Summary: (.+)", text)
        sentiment = re.search(r"Sentiment: (\\w+)", text, re.IGNORECASE)
        action_items = re.findall(r"- (.+)", text)
        return {
            "summary": summary.group(1) if summary else None,
            "sentiment": sentiment.group(1) if sentiment else None,
            "action_items": action_items
        }

parser = SupportSummaryParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed support summary:", result) 