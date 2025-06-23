"""
Streamlit Chatbot with LangChain Prompts and Open Source Models
"""

import streamlit as st
from model_setup import get_model, MODEL_CONFIGS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="LangChain Open Source Chatbot", page_icon="🤖")
st.title("🤖 LangChain Open Source Chatbot")
st.markdown("""
A simple, dynamic chatbot using open source models (Ollama, Hugging Face) and LangChain prompts.
- **Select your model**
- **Type your message**
- **See the conversation history**
""")

# Sidebar: Model selection
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Model Provider",
    options=["ollama", "huggingface"],
    index=0
)

if model_type == "ollama":
    model_options = list(MODEL_CONFIGS["ollama"].keys())
else:
    model_options = list(MODEL_CONFIGS["huggingface"].keys())

model_name = st.sidebar.selectbox(
    "Model Name",
    options=model_options,
    index=0
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 64, 2048, 512, 32)

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input
user_input = st.chat_input("Type your message and press Enter...")

# Display conversation history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Handle user input
if user_input:
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Build chat prompt (dynamic, includes history)
    chat_history = ""
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            chat_history += f"User: {msg['content']}\n"
        else:
            chat_history += f"Assistant: {msg['content']}\n"

    # Use a simple chat template
    chat_template = PromptTemplate(
        input_variables=["history", "input"],
        template="""
        The following is a conversation between a helpful AI assistant and a user.
        {history}
        User: {input}
        Assistant:
        """
    )
    prompt = chat_template.format(history=chat_history, input=user_input)

    # Get the model
    llm = get_model(model_type, model_name, temperature=temperature, max_tokens=max_tokens)

    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception as e:
            response_text = f"[Error: {e}]"

    # Add assistant message to history
    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)

st.markdown("---")
st.markdown("Built with [LangChain](https://python.langchain.com/) and [Streamlit](https://streamlit.io/). Supports open source models via Ollama and Hugging Face.") 