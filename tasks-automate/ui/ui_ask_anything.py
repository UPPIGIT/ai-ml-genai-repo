import streamlit as st
from prompts import get_prompt, PROMPT_VERSIONS

def render_ask_anything_ui(provider_choice, model_choice, call_llm):
    st.markdown("""
        <style>
        .ask-header {font-size:2.2rem; font-weight:700; color:#10b981; margin-bottom:0.2em;}
        .ask-sub {font-size:1.1rem; color:#64748B; margin-bottom:1.2em;}
        .ask-section {
            margin-top:1.5em; margin-bottom:0.5em; font-size:1.1rem; color:#0F172A; font-weight:600;
            background: linear-gradient(90deg, #d1fae5 0%, #f0fdf4 100%);
            padding: 0.5em 1em; border-radius: 8px; box-shadow: 0 2px 8px #bbf7d044;
        }
        .ask-answer-box {
            background: #f0fdf4;
            border-radius: 8px;
            box-shadow: 0 2px 8px #bbf7d044;
            padding: 1em 1.2em;
            margin-bottom: 1em;
            color: #0F172A;
            font-size: 1.08rem;
            font-weight: 500;
            word-break: break-word;
        }
        .ask-footer {margin-top:2em; font-size:0.95rem; color:#64748B; text-align:center;}
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="ask-header">💬 Ask Anything (AI Assistant)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ask-sub">Ask any question and get a simple, crisp, conversational answer from AI.<br><b>Provider:</b> {provider_choice} &nbsp; <b>Model:</b> {model_choice}</div>', unsafe_allow_html=True)
    st.caption(f"Model: {model_choice}")

    prompt_version = st.selectbox("Prompt version", list(PROMPT_VERSIONS['ask_anything'].keys()), format_func=lambda v: PROMPT_VERSIONS['ask_anything'][v])

    st.markdown('<div class="ask-section">Your Question</div>', unsafe_allow_html=True)
    question = st.text_area("Type your question here:", height=160)

    def build_prompt():
        prompt_template = get_prompt('ask_anything', prompt_version)
        return prompt_template.format(question=question)

    st.markdown('<div class="ask-section">Get Your Answer</div>', unsafe_allow_html=True)
    if st.button("🤖 Ask", use_container_width=True):
        with st.spinner("Thinking..."):
            st.session_state["ask_anything_answer"] = call_llm(provider_choice, model_choice, build_prompt())
    if "ask_anything_answer" in st.session_state:
        st.subheader("Answer:")
        st.markdown(f"<div class='ask-answer-box'>{st.session_state['ask_anything_answer']}</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <button onclick="navigator.clipboard.writeText(document.querySelector('pre') ? document.querySelector('pre').innerText : document.querySelector('.ask-answer-box').innerText)">📋 Copy to clipboard</button>
            """,
            unsafe_allow_html=True
        )
        st.caption("Select and copy, or use the button above.")
        if st.button("🔄 Regenerate", key="regenerate_ask_anything", use_container_width=True):
            with st.spinner("Regenerating..."):
                st.session_state["ask_anything_answer"] = call_llm(provider_choice, model_choice, build_prompt())
    st.markdown('<div class="ask-footer">Need help? <a href="https://streamlit.io" target="_blank">Learn more about Streamlit</a></div>', unsafe_allow_html=True) 