import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import asyncio

# Load environment variables from .env file
load_dotenv()

def get_llm(provider_choice, model_choice):
    """
    Returns a LangChain LLM instance for the given provider and model.
    provider_choice: 'Gemini', 'Groq', or 'HuggingFace'
    model_choice: model name as selected in the sidebar
    """
    if provider_choice == "Gemini":
        # Map sidebar model names to actual Gemini model IDs
        model_map = {
            "Gemini Pro": "gemini-pro",
            "Gemini-1.5-Flash": "gemini-1.5-flash",
            "Gemini-2.0-Flash": "gemini-2.0-flash"
        }
        model_id = model_map.get(model_choice, "gemini-pro")
        return ChatGoogleGenerativeAI(model=model_id)
    elif provider_choice == "Groq":
        model_map = {
            "Llama-3-Groq-70B-Tool-Use": "llama3-70b-8192-tool-use",
            "Llama-3-70B-Instruct": "llama3-70b-8192",
            "Llama-3-Groq-8B-Tool-Use": "llama3-8b-8192-tool-use",
            "Llama-3-8B-Instruct": "llama3-8b-8192"
        }
        model_id = model_map.get(model_choice, "llama3-70b-8192")
        return ChatGroq(model=model_id)
    elif provider_choice == "HuggingFace":
        model_map = {
            "mistralai/Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
            "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"
        }
        repo_id = model_map.get(model_choice, "mistralai/Mistral-7B-Instruct-v0.2")
        llm = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation")
        return ChatHuggingFace(llm=llm)
    else:
        raise ValueError(f"Unknown provider/model: {provider_choice} / {model_choice}")

async def async_call_llm(provider_choice, model_choice, prompt):
    llm = get_llm(provider_choice, model_choice)
    return (await llm.ainvoke(prompt)).content

async def async_batch_llm(provider_choice, model_choice, prompts):
    llm = get_llm(provider_choice, model_choice)
    results = await llm.abatch(prompts)
    return [r.content for r in results] 