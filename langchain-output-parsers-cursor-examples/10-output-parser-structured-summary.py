"""
Example 10: Structured Summary Output Parser (with Gemini)
This example demonstrates extracting structured information from a review using a custom parser.
"""
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
prompt = PromptTemplate.from_template(
    "Summarize the following review. Extract: key points (as a list), sentiment (positive/negative/neutral), and any action items.\n\nReview: The app is fast and easy to use, but sometimes crashes unexpectedly. Please fix the bugs."
)

class StructuredSummaryParser(BaseOutputParser):
    def parse(self, text: str):
        # Simple regex-based extraction for demonstration
        key_points = re.findall(r'- (.+)', text)
        sentiment_match = re.search(r'Sentiment: (\w+)', text, re.IGNORECASE)
        action_items = re.findall(r'Action Item: (.+)', text)
        return {
            "key_points": key_points,
            "sentiment": sentiment_match.group(1) if sentiment_match else None,
            "action_items": action_items
        }

parser = StructuredSummaryParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed structured summary:", result) 