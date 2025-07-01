import streamlit as st
import pandas as pd
from llm_providers import get_llm
from ui.ui_email import render_email_ui
from ui.ui_markdown import render_markdown_ui
from ui.ui_qa import render_qa_ui
from ui.ui_effort import render_effort_ui
from ui.ui_ask_anything import render_ask_anything_ui

st.set_page_config(page_title="AI Task Automator", layout="wide")

st.title("🧠 AI Task Automator")

# Sidebar navigation
st.sidebar.title("Navigation")

PROVIDER_MODELS = {
    "Gemini": ["Gemini Pro", "Gemini-1.5-Flash", "Gemini-2.0-Flash"],
    "Groq": [
        "Llama-3-Groq-70B-Tool-Use",
        "Llama-3-70B-Instruct",
        "Llama-3-Groq-8B-Tool-Use",
        "Llama-3-8B-Instruct"
    ],
    "HuggingFace": [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
}

provider_choice = st.sidebar.selectbox("Select Provider:", list(PROVIDER_MODELS.keys()))
model_choice = st.sidebar.selectbox("Select Model:", PROVIDER_MODELS[provider_choice])

section = st.sidebar.radio(
    "Go to:",
    [
        "Email Rewriting Assistant",
        "Markdown Content Generator",
        "Question-Answer Assistant",
        "Effort Estimation Generator",
        "Ask Anything (AI Assistant)"
    ]
)

def call_llm(provider_choice, model_choice, prompt):
    llm = get_llm(provider_choice, model_choice)
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"[LLM error: {e}]"

if section == "Email Rewriting Assistant":
    render_email_ui(provider_choice, model_choice, call_llm)
elif section == "Markdown Content Generator":
    render_markdown_ui(provider_choice, model_choice, call_llm)
elif section == "Question-Answer Assistant":
    render_qa_ui(provider_choice, model_choice, call_llm)
elif section == "Effort Estimation Generator":
    render_effort_ui(provider_choice, model_choice, call_llm)
elif section == "Ask Anything (AI Assistant)":
    render_ask_anything_ui(provider_choice, model_choice, call_llm) 