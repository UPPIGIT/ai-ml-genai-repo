"""
6_youtube_transcript_loader.py
-----------------------------
This script demonstrates loading a YouTube video transcript and using an LLM to answer a question about the video content.
"""

from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key)
load_dotenv()

# YouTube video URL (replace with any public video with captions)
youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Load the transcript
documents = YoutubeLoader.from_youtube_url(youtube_url).load()

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0)

# Create a QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Ask a question about the video
question = "What is this video about?"

# Run the QA chain
answer = qa_chain.run(input_documents=documents, question=question)

print("Question:", question)
print("Answer:", answer) 