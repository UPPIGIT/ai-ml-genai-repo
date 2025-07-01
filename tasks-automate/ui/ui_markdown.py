import streamlit as st
from prompts import PROMPT_VERSIONS, get_prompt
from llm_providers import async_call_llm, async_batch_llm
import asyncio

def render_markdown_ui(provider_choice, model_choice, call_llm):
    st.markdown("""
        <style>
        .md-header {font-size:2.2rem; font-weight:700; color:#6366F1; margin-bottom:0.2em;}
        .md-sub {font-size:1.1rem; color:#64748B; margin-bottom:1.2em;}
        .md-section {
            margin-top:1.5em; margin-bottom:0.5em; font-size:1.1rem; color:#0F172A; font-weight:600;
            background: linear-gradient(90deg, #e0e7ff 0%, #f3f4f6 100%);
            padding: 0.5em 1em; border-radius: 8px; box-shadow: 0 2px 8px #e0e7ff44;
        }
        .md-footer {margin-top:2em; font-size:0.95rem; color:#64748B; text-align:center;}
        .md-answer-box {
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
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="md-header">📝 Markdown Content Generator</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="md-sub">Convert technical/conceptual content into well-structured Markdown.<br><b>Provider:</b> {provider_choice} &nbsp; <b>Model:</b> {model_choice}</div>', unsafe_allow_html=True)
    st.caption(f"Model: {model_choice}")

    st.markdown('<div class="md-section">1️⃣ Settings</div>', unsafe_allow_html=True)
    prompt_version = st.selectbox("Prompt version", list(PROMPT_VERSIONS['markdown'].keys()), format_func=lambda v: PROMPT_VERSIONS['markdown'][v])
    batch_mode = st.checkbox("Batch Mode (process multiple markdown generations at once)")

    st.markdown('<div class="md-section">2️⃣ Content Input</div>', unsafe_allow_html=True)
    if batch_mode:
        topics = st.text_area("Paste your topics/titles (one per line):")
        contents = st.text_area("Paste your contents (one per line, matching topics):")
    else:
        topic = st.text_input("Topic or Title:")
        content = st.text_area("Paste your content:")
    add_examples = st.checkbox("Add example section (optional)")
    instructions_md = st.text_area("Additional instructions (optional):")

    def build_md_prompt(content):
        prompt_template = get_prompt('markdown', prompt_version)
        prompt = prompt_template.format(
            instructions=f" {instructions_md}" if instructions_md else "",
            content=content
        )
        if add_examples:
            prompt += "\n\nAdd an example section."
        return prompt

    # Prompt preview and editing
    if batch_mode:
        content_list = [c.strip() for c in contents.splitlines() if c.strip()]
        if content_list:
            preview_prompt = build_md_prompt(content_list[0])
            with st.expander("Preview/Edit Prompt (first content)", expanded=False):
                edited_prompt = st.text_area("Prompt to send (first content)", preview_prompt, key="edit_prompt_batch_md")
                use_edited = st.checkbox("Use edited prompt for all markdowns (overrides template)", key="use_edited_batch_md")
        else:
            edited_prompt = None
            use_edited = False
    else:
        preview_prompt = build_md_prompt(content) if content else ""
        with st.expander("Preview/Edit Prompt", expanded=False):
            edited_prompt = st.text_area("Prompt to send", preview_prompt, key="edit_prompt_single_md")
            use_edited = st.checkbox("Use edited prompt (overrides template)", key="use_edited_single_md")

    st.markdown('<div class="md-section">3️⃣ Generate Markdown</div>', unsafe_allow_html=True)
    if batch_mode:
        if st.button("📝 Generate All Markdown", use_container_width=True):
            content_list = [c.strip() for c in contents.splitlines() if c.strip()]
            results = []
            errors = []
            progress = st.progress(0)
            total = len(content_list)
            for i, c in enumerate(content_list):
                try:
                    prompt = edited_prompt if use_edited and edited_prompt else build_md_prompt(c)
                    result = asyncio.run(async_call_llm(provider_choice, model_choice, prompt))
                    results.append(result)
                    errors.append(None)
                except Exception as e:
                    results.append(None)
                    errors.append(str(e))
                progress.progress((i + 1) / total)
            progress.empty()
            for i, (c, md, error) in enumerate(zip(content_list, results, errors), 1):
                st.markdown(f"**Markdown {i}:**")
                if md:
                    st.markdown(f"<div class='md-answer-box'>{md}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <button onclick=\"navigator.clipboard.writeText(document.querySelectorAll('pre')[{i-1}].innerText)\">📋 Copy to clipboard</button>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption("Select and copy, or use the button above.")
                else:
                    st.error(f"Failed: {error}")
            st.success(f"Batch complete: {sum(r is not None for r in results)} succeeded, {sum(r is None for r in results)} failed.")
    else:
        def build_single_prompt():
            if use_edited and edited_prompt:
                return edited_prompt
            return build_md_prompt(content)
        if st.button("📝 Generate Markdown", use_container_width=True):
            with st.spinner("Generating markdown..."):
                st.session_state["markdown_result"] = call_llm(provider_choice, model_choice, build_single_prompt())
        if "markdown_result" in st.session_state:
            st.subheader("Markdown Output:")
            st.markdown(f"<div class='md-answer-box'>{st.session_state['markdown_result']}</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <button onclick=\"navigator.clipboard.writeText(document.querySelector('pre').innerText)\">📋 Copy to clipboard</button>
                """,
                unsafe_allow_html=True
            )
            st.caption("Select and copy, or use the button above.")
            if st.button("🔄 Regenerate", key="regenerate_md", use_container_width=True):
                with st.spinner("Regenerating..."):
                    st.session_state["markdown_result"] = call_llm(provider_choice, model_choice, build_single_prompt())
    st.markdown('<div class="md-footer">Need help? <a href="https://streamlit.io" target="_blank">Learn more about Streamlit</a></div>', unsafe_allow_html=True) 