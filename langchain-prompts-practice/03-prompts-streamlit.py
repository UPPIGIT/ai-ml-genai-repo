import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="LangChain Open Source Chatbot",page_icon="🤖")
st.title("LangChain Open Source Chatbot 🤖")

st.markdown("""
A simple, dynamic chatbot using open source models (Gemini
, Hugging Face) and LangChain prompts.
- **Select your model**
- **Type your message**
- **See the conversation history**
""")

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    options=["Gemini", "Hugging Face"],
    index=0
)

if model_choice == "Gemini":
    model_options = ['gemini-1.5-flash', 'gemini-2.5-flash-lite-preview-06-17']
else:
    model_options = ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-70B-Instruct']


model_name = st.sidebar.selectbox(
    "Select a model",
    options=model_options,
    index=0
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 2000, 500, 50)

# Session state for conversation
if 'messages' not in st.session_state:
    st.session_state['messages']= []

#chat input
user_input = st.chat_input("Type your message and presee Enter ...")

for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.chat_message("user").write(msg['content'])
    else:
        st.chat_message("assistant").write(msg['content'])

if user_input:

    #add user message to session state
    st.session_state['messages'].append({"role": "user", "content": user_input})

    #Build chat history
    chat_history = ""
    for msg in st.session_state['messages']:
        if msg['role'] == 'user':
            chat_history += f"User: {msg['content']}\n"
        else:
            chat_history += f"Assistant: {msg['content']}\n"

    chat_template = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template="""
                The following is a conversation between a helpful AI assistant and a user.
                {history}
                User: {input}
                Assistant:
                """
    )

    prompt = chat_template.format(
        history=chat_history,
        input=user_input
    )

    if model_choice == "Gemini":
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    else:
        llm=HuggingFaceEndpoint(repo_id=model_name,
                               task="text-generation")
        

        model = ChatHuggingFace(
            llm=llm,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
  

    #generate response

    with st.spinner("Thinking ...."):
        try :
            response = model.invoke(prompt)
            if hasattr(response,'content'):
                response_content = response.content
            else:
                response_content = response_text = str(response)
        except Exception as e:
            #st.error(f"Error generating response: {e}")
            response_content = "Sorry, I couldn't process your request."
    
    st.session_state['messages'].append({"role": "assistant", "content": response_content})
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response_content)
    print("chat_history:", chat_history)

st.markdown("---")
st.markdown("Built with [LangChain](https://python.langchain.com/) and [Streamlit](https://streamlit.io/). Supports open source models via Ollama and Hugging Face.") 