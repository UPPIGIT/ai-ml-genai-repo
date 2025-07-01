import streamlit as st
from prompts import PROMPT_VERSIONS, get_prompt
from llm_providers import async_call_llm, async_batch_llm
import asyncio

def render_qa_ui(provider_choice, model_choice, call_llm):
    st.markdown("""
        <style>
        .qa-header {font-size:2.2rem; font-weight:700; color:#6366F1; margin-bottom:0.2em;}
        .qa-sub {font-size:1.1rem; color:#64748B; margin-bottom:1.2em;}
        .qa-section {
            margin-top:1.5em; margin-bottom:0.5em; font-size:1.1rem; color:#0F172A; font-weight:600;
            background: linear-gradient(90deg, #e0e7ff 0%, #f3f4f6 100%);
            padding: 0.5em 1em; border-radius: 8px; box-shadow: 0 2px 8px #e0e7ff44;
        }
        .qa-footer {margin-top:2em; font-size:0.95rem; color:#64748B; text-align:center;}
        .qa-answer-box {
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
        .qa-batch-summary {
            background: #fef9c3;
            border-radius: 8px;
            box-shadow: 0 2px 8px #fde68a44;
            padding: 0.7em 1em;
            margin-bottom: 1em;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="qa-header">🤖 Question-Answer Assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="qa-sub">Ask questions and get AI-generated answers. Optionally, evaluate the answer quality.<br><b>Provider:</b> {provider_choice} &nbsp; <b>Model:</b> {model_choice}</div>', unsafe_allow_html=True)
    st.caption(f"Model: {model_choice}")

    st.markdown('<div class="qa-section">1️⃣ Settings</div>', unsafe_allow_html=True)
    cols = st.columns([2,1,1])
    prompt_version = cols[0].selectbox("Prompt version", list(PROMPT_VERSIONS['qa_eval'].keys()), format_func=lambda v: PROMPT_VERSIONS['qa_eval'][v])
    batch_mode = cols[1].checkbox("Batch Mode (multiple QAs)")
    evaluate = cols[2].checkbox("Evaluate answer quality (experimental)")

    st.markdown('<div class="qa-section">2️⃣ Your Question(s)</div>', unsafe_allow_html=True)
    if batch_mode:
        questions = st.text_area("Paste your questions (one per line):", height=160)
    else:
        question = st.text_area("Your question:", height=160)

    def build_qa_prompt(q):
        prompt_template = get_prompt('qa_eval', prompt_version)
        return prompt_template.format(question=q, options="", instructions="")
    def build_eval_prompt(q, a):
        prompt_template = get_prompt('qa_eval', prompt_version)
        return prompt_template.format(question=q, options="", instructions="\nAnswer:\n" + a)

    # Prompt preview and editing
    if batch_mode:
        question_list = [q.strip() for q in questions.splitlines() if q.strip()]
        if question_list:
            preview_prompt = build_qa_prompt(question_list[0])
            with st.expander("Preview/Edit Prompt (first question)", expanded=False):
                edited_prompt = st.text_area("Prompt to send (first question)", preview_prompt, key="edit_prompt_batch_qa")
                use_edited = st.checkbox("Use edited prompt for all QAs (overrides template)", key="use_edited_batch_qa")
        else:
            edited_prompt = None
            use_edited = False
    else:
        preview_prompt = build_qa_prompt(question) if question else ""
        with st.expander("Preview/Edit Prompt", expanded=False):
            edited_prompt = st.text_area("Prompt to send", preview_prompt, key="edit_prompt_single_qa")
            use_edited = st.checkbox("Use edited prompt (overrides template)", key="use_edited_single_qa")

    st.markdown('<div class="qa-section">3️⃣ Run Q&A</div>', unsafe_allow_html=True)
    if batch_mode:
        if st.button("🚀 Get Answers for All", use_container_width=True):
            question_list = [q.strip() for q in questions.splitlines() if q.strip()]
            results = []
            errors = []
            progress = st.progress(0)
            total = len(question_list)
            for i, q in enumerate(question_list):
                try:
                    prompt = edited_prompt if use_edited and edited_prompt else build_qa_prompt(q)
                    answer = asyncio.run(async_call_llm(provider_choice, model_choice, prompt))
                    results.append(answer)
                    errors.append(None)
                except Exception as e:
                    results.append(None)
                    errors.append(str(e))
                progress.progress((i + 1) / total)
            progress.empty()
            for i, (q, a, error) in enumerate(zip(question_list, results, errors), 1):
                st.markdown(f"<div class='qa-section' style='background:#f3f4f6;'>Q{i}: {q}</div>", unsafe_allow_html=True)
                if a:
                    st.markdown(f"<div class='qa-answer-box'>{a}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <button onclick=\"navigator.clipboard.writeText(document.querySelectorAll('pre')[{i-1}].innerText)\">📋 Copy to clipboard</button>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption("Select and copy, or use the button above.")
                    if evaluate:
                        with st.spinner("Evaluating answer..."):
                            eval_result = call_llm(provider_choice, model_choice, build_eval_prompt(q, a))
                        st.info(f"Evaluation: {eval_result}")
                else:
                    st.error(f"Failed: {error}")
            st.markdown(f"<div class='qa-batch-summary'><b>Batch complete:</b> {sum(r is not None for r in results)} succeeded, {sum(r is None for r in results)} failed.</div>", unsafe_allow_html=True)
    else:
        def build_single_prompt():
            if use_edited and edited_prompt:
                return edited_prompt
            return build_qa_prompt(question)
        if st.button("💡 Get Answer", use_container_width=True):
            with st.spinner("Getting answer..."):
                st.session_state["qa_answer"] = call_llm(provider_choice, model_choice, build_single_prompt())
        if "qa_answer" in st.session_state:
            st.subheader("Answer:")
            st.markdown(f"<div class='qa-answer-box'>{st.session_state['qa_answer']}</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <button onclick=\"navigator.clipboard.writeText(document.querySelector('pre').innerText)\">📋 Copy to clipboard</button>
                """,
                unsafe_allow_html=True
            )
            st.caption("Select and copy, or use the button above.")
            if evaluate:
                with st.spinner("Evaluating answer..."):
                    eval_result = call_llm(provider_choice, model_choice, build_eval_prompt(question, st.session_state["qa_answer"]))
                st.info(f"Evaluation: {eval_result}")
            if st.button("🔄 Regenerate", key="regenerate_qa", use_container_width=True):
                with st.spinner("Regenerating..."):
                    st.session_state["qa_answer"] = call_llm(provider_choice, model_choice, build_single_prompt())
    st.markdown('<div class="qa-footer">Need help? <a href="https://streamlit.io" target="_blank">Learn more about Streamlit</a></div>', unsafe_allow_html=True) 