import streamlit as st
from prompts import PROMPT_VERSIONS, get_prompt
from llm_providers import async_call_llm, async_batch_llm
import asyncio

def render_email_ui(provider_choice, model_choice, call_llm):
    st.markdown("""
        <style>
        .email-header {font-size:2.2rem; font-weight:700; color:#F59E42; margin-bottom:0.2em;}
        .email-sub {font-size:1.1rem; color:#64748B; margin-bottom:1.2em;}
        .email-section {
            margin-top:1.5em; margin-bottom:0.5em; font-size:1.1rem; color:#0F172A; font-weight:600;
            background: linear-gradient(90deg, #fef3c7 0%, #f3f4f6 100%);
            padding: 0.5em 1em; border-radius: 8px; box-shadow: 0 2px 8px #fde68a44;
        }
        .email-footer {margin-top:2em; font-size:0.95rem; color:#64748B; text-align:center;}
        .email-answer-box {
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
    st.markdown('<div class="email-header">📧 Email Rewriting Assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="email-sub">Rewrite emails in your chosen tone using AI. Paste your email, select a tone, and get a professional rewrite.<br><b>Provider:</b> {provider_choice} &nbsp; <b>Model:</b> {model_choice}</div>', unsafe_allow_html=True)

    st.markdown('<div class="email-section">1️⃣ Settings</div>', unsafe_allow_html=True)
    cols = st.columns([2,1,1])
    prompt_version = cols[0].selectbox("Prompt version", list(PROMPT_VERSIONS['email'].keys()), format_func=lambda v: PROMPT_VERSIONS['email'][v])
    tone = cols[1].selectbox("Select tone:", ["Professional", "Formal", "Simple", "Other (custom)"])
    batch_mode = cols[2].checkbox("Batch Mode (multiple emails)")

    custom_tone = ""
    if tone == "Other (custom)":
        custom_tone = st.text_input("Specify custom tone:")

    st.markdown('<div class="email-section">2️⃣ Email Content</div>', unsafe_allow_html=True)
    if batch_mode:
        raw_emails = st.text_area("Paste your raw emails (one per line):", height=120)
    else:
        raw_email = st.text_area("Paste your raw email content:", height=120)

    instructions = st.text_area("Additional instructions (optional):", height=80)

    def build_email_prompt(email_content):
        prompt_template = get_prompt('email', prompt_version)
        prompt_filled = prompt_template.format(
            tone=custom_tone or tone,
            instructions=f" {instructions}" if instructions else "",
            email_content=email_content
        )
        return prompt_filled

    # Prompt preview and editing
    if batch_mode:
        emails = [e.strip() for e in raw_emails.splitlines() if e.strip()]
        if emails:
            preview_prompt = build_email_prompt(emails[0])
            with st.expander("Preview/Edit Prompt (first email)", expanded=False):
                edited_prompt = st.text_area("Prompt to send (first email)", preview_prompt, key="edit_prompt_batch")
                use_edited = st.checkbox("Use edited prompt for all emails (overrides template)", key="use_edited_batch")
        else:
            edited_prompt = None
            use_edited = False
    else:
        preview_prompt = build_email_prompt(raw_email) if raw_email else ""
        with st.expander("Preview/Edit Prompt", expanded=False):
            edited_prompt = st.text_area("Prompt to send", preview_prompt, key="edit_prompt_single")
            use_edited = st.checkbox("Use edited prompt (overrides template)", key="use_edited_single")

    st.markdown('<div class="email-section">3️⃣ Run Email Rewriting</div>', unsafe_allow_html=True)
    if batch_mode:
        if st.button("🚀 Rewrite All Emails", use_container_width=True):
            emails = [e.strip() for e in raw_emails.splitlines() if e.strip()]
            results = []
            errors = []
            progress = st.progress(0)
            total = len(emails)
            for i, email in enumerate(emails):
                try:
                    prompt = edited_prompt if use_edited and edited_prompt else build_email_prompt(email)
                    result = asyncio.run(async_call_llm(provider_choice, model_choice, prompt))
                    results.append(result)
                    errors.append(None)
                except Exception as e:
                    results.append(None)
                    errors.append(str(e))
                progress.progress((i + 1) / total)
            progress.empty()
            for i, (email, rewritten, error) in enumerate(zip(emails, results, errors), 1):
                st.markdown(f"**Email {i}:**")
                if rewritten:
                    st.markdown(f"<div class='email-answer-box'>{rewritten}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <button onclick=\"navigator.clipboard.writeText(document.querySelectorAll('pre')[{i-1}].innerText)\">📋 Copy to clipboard</button>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption("Select and copy, or use the button above.")
                else:
                    st.error(f"Failed: {error}")
            st.markdown(f"<div style='background:#DCFCE7;padding:0.7em 1em;border-radius:8px;margin-bottom:1em;'><b>Batch complete:</b> {sum(r is not None for r in results)} succeeded, {sum(r is None for r in results)} failed.</div>", unsafe_allow_html=True)
    else:
        def build_single_prompt():
            if use_edited and edited_prompt:
                return edited_prompt
            return build_email_prompt(raw_email)
        if st.button("✉️ Rewrite Email", use_container_width=True):
            with st.spinner("Rewriting..."):
                st.session_state["rewritten_email"] = call_llm(provider_choice, model_choice, build_single_prompt())
        if "rewritten_email" in st.session_state:
            st.subheader("Rewritten Email:")
            st.markdown(f"<div class='email-answer-box'>{st.session_state['rewritten_email']}</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <button onclick=\"navigator.clipboard.writeText(document.querySelector('pre').innerText)\">📋 Copy to clipboard</button>
                """,
                unsafe_allow_html=True
            )
            st.caption("Select and copy, or use the button above.")
            if st.button("🔄 Regenerate", key="regenerate_email", use_container_width=True):
                with st.spinner("Regenerating..."):
                    st.session_state["rewritten_email"] = call_llm(provider_choice, model_choice, build_single_prompt())
    st.markdown('<div class="email-footer">Need help? <a href="https://streamlit.io" target="_blank">Learn more about Streamlit</a></div>', unsafe_allow_html=True) 