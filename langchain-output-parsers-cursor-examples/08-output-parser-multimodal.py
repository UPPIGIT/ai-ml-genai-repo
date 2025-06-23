"""
Example 8: Multimodal Output Parser (Text and Image URLs, with Gemini)
This example demonstrates parsing Gemini output that includes both text and image URLs.
"""
from langchain.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
prompt = PromptTemplate.from_template(
    "Describe the Eiffel Tower and provide a direct image URL."
)

class TextImageParser(StrOutputParser):
    def parse(self, text: str):
        # Extract image URL (simple regex for demonstration)
        url_match = re.search(r'(https?://\S+\.(?:jpg|jpeg|png|gif))', text)
        url = url_match.group(1) if url_match else None
        description = text.split('http')[0].strip() if url else text.strip()
        return {"description": description, "image_url": url}

parser = TextImageParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed multimodal output:", result) 