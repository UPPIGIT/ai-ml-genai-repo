import streamlit as st
import pandas as pd
from prompts import PROMPT_VERSIONS, get_prompt
from llm_providers import async_call_llm
import asyncio
from io import BytesIO, StringIO
import re
import hashlib
import uuid

def normalize_effort_columns(df):
    if df is None:
        return None
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'categor' in col_lower:
            col_map[col] = 'Category'
        elif 'activit' in col_lower:
            col_map[col] = 'Activity'
        elif 'hour' in col_lower:
            col_map[col] = 'Hours'
    df = df.rename(columns=col_map)
    return df

def has_required_columns(header_cols):
    required = {'category', 'activity', 'hours'}
    found = set([c.lower().strip() for c in header_cols])
    return required.issubset(found)

def extract_table_from_response(response):
    # Try to find a CSV table (at least 2 lines with commas)
    csv_table = re.search(r'([\w\s]+,[\w\s]+,[\w\s]+\n(?:.+,.+,.+\n)+)', response)
    if csv_table:
        table_str = csv_table.group(0)
        try:
            df = pd.read_csv(StringIO(table_str))
            df = normalize_effort_columns(df)
            df._debug_csv_str = table_str
            return {'df': df, 'debug_csv': table_str, 'header_cols': list(df.columns)}
        except Exception:
            return {'df': None, 'debug_csv': table_str, 'header_cols': []}
    # Fallback to markdown table parsing if CSV not found
    md_table = re.search(r'(\|\s*:?[-]+:?.*\n(?:\|.*\n)+)', response)
    if md_table:
        table_str = md_table.group(0)
        lines = [line for line in table_str.strip().splitlines() if line.strip()]
        def is_separator(line):
            return bool(re.match(r'^\|?\s*:?[- ]+:?\s*\|.*$', line))
        header_idx = next((i for i, line in enumerate(lines) if not is_separator(line)), None)
        if header_idx is not None:
            header = lines[header_idx].strip('|').strip()
            header_cols = [col.strip() for col in header.split('|')]
            if not has_required_columns(header_cols):
                csv_str = '\n'.join(lines)
                df = None
                df_debug = csv_str
                df_header_cols = header_cols
                return {'df': df, 'debug_csv': df_debug, 'header_cols': df_header_cols}
            data_lines = [l for i, l in enumerate(lines) if i != header_idx and not is_separator(l)]
            cleaned_data = []
            for l in data_lines:
                row = l.strip('|').strip()
                cols = [col.strip() for col in row.split('|')]
                if len(cols) < len(header_cols):
                    cols += [''] * (len(header_cols) - len(cols))
                elif len(cols) > len(header_cols):
                    cols = cols[:len(header_cols)]
                cleaned_data.append(cols)
            csv_str = ','.join(header_cols) + '\n'
            for row in cleaned_data:
                csv_str += ','.join(row) + '\n'
            try:
                df = pd.read_csv(StringIO(csv_str))
                df = normalize_effort_columns(df)
                df._debug_csv_str = csv_str
                return {'df': df, 'debug_csv': csv_str, 'header_cols': header_cols}
            except Exception:
                return {'df': None, 'debug_csv': csv_str, 'header_cols': header_cols}
    try:
        df = pd.read_csv(StringIO(response))
        df = normalize_effort_columns(df)
        df._debug_csv_str = response
        return {'df': df, 'debug_csv': response, 'header_cols': list(df.columns)}
    except Exception:
        return {'df': None, 'debug_csv': response, 'header_cols': []}

def sum_hours_column(df):
    if df is None or 'Hours' not in df.columns:
        return None
    try:
        return int(df['Hours'].astype(float).sum())
    except Exception:
        return None

def clamp_and_adjust_hours(df, total_hours, min_hour=1, max_hour=40):
    if df is None or 'Hours' not in df.columns:
        return df, False, 'No Hours column.'
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce').fillna(0).round().astype(int)
    df['Hours'] = df['Hours'].clip(lower=min_hour, upper=max_hour)
    sum_hours = df['Hours'].sum()
    changed = False
    if sum_hours != total_hours:
        diff = total_hours - sum_hours
        if len(df) == 0:
            return df, False, 'No activities to adjust.'
        # Adjust the largest or smallest value
        if diff > 0:
            idx = df['Hours'].idxmax()
            df.at[idx, 'Hours'] += diff
        else:
            idx = df['Hours'].idxmax()
            if df.at[idx, 'Hours'] + diff >= min_hour:
                df.at[idx, 'Hours'] += diff
            else:
                # Try to distribute the reduction
                for i in df.sort_values('Hours', ascending=False).index:
                    reducible = df.at[i, 'Hours'] - min_hour
                    take = min(-diff, reducible)
                    df.at[i, 'Hours'] -= take
                    diff += take
                    if diff == 0:
                        break
        changed = True
    sum_hours = df['Hours'].sum()
    if sum_hours != total_hours:
        return df, False, f'Unable to adjust hours to match total ({sum_hours} vs {total_hours}) after clamping.'
    return df, changed, None

def render_effort_ui(provider_choice, model_choice, call_llm):
    st.markdown("""
        <style>
        .effort-header {font-size:2.2rem; font-weight:700; color:#3B82F6; margin-bottom:0.2em;}
        .effort-sub {font-size:1.1rem; color:#64748B; margin-bottom:1.2em;}
        .effort-section {
            margin-top:1.5em; margin-bottom:0.5em; font-size:1.1rem; color:#0F172A; font-weight:600;
            background: linear-gradient(90deg, #e0e7ff 0%, #f3f4f6 100%);
            padding: 0.5em 1em; border-radius: 8px; box-shadow: 0 2px 8px #e0e7ff44;
        }
        .effort-footer {margin-top:2em; font-size:0.95rem; color:#64748B; text-align:center;}
        .effort-answer-box {
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
    st.markdown('<div class="effort-header">📊 Effort Estimation from CSV</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="effort-sub">Distribute project hours across activities using AI. Upload your CSV, describe your story, and get a professional allocation.<br><b>Provider:</b> {provider_choice} &nbsp; <b>Model:</b> {model_choice}</div>', unsafe_allow_html=True)
    st.caption(f"Model: {model_choice}")

    with st.container():
        st.markdown('<div class="effort-section">1️⃣ Project Details</div>', unsafe_allow_html=True)
        cols = st.columns([2,1,2])
        main_story = cols[0].text_input("Main Story/Task:")
        total_hours = cols[1].number_input("Total Hours to Distribute:", min_value=1, value=40)
        background_details = cols[2].text_area("Task Background Details (optional):", height=80)

    with st.container():
        st.markdown('<div class="effort-section">2️⃣ Upload Activities CSV</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file (Activity, Category)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df, height=180)
            cols2 = st.columns(2)
            activity_col = cols2[0].selectbox("Select the column for Activity:", df.columns, key="activity_col")
            category_col = cols2[1].selectbox("Select the column for Category:", df.columns, key="category_col")
            csv_activities = df[[activity_col, category_col]].to_csv(index=False)
        else:
            activity_col = category_col = csv_activities = None

    with st.container():
        st.markdown('<div class="effort-section">3️⃣ AI Prompt & Settings</div>', unsafe_allow_html=True)
        cols3 = st.columns([2,1,1])
        prompt_version = cols3[0].selectbox("Prompt version", list(PROMPT_VERSIONS['effort'].keys()), format_func=lambda v: PROMPT_VERSIONS['effort'][v])
        instructions = cols3[1].text_area("Additional instructions (optional):", height=80)
        max_retries = cols3[2].number_input("Max adjustment retries:", min_value=1, max_value=10, value=3)

    if uploaded_file:
        def build_effort_prompt():
            background_section = f" with the following background details: '{background_details}'" if background_details else ""
            return (
                f"You are a project manager. Given the main story/task: '{main_story}'{background_section} and a list of activities with categories, select only the most suitable and required activities for this story. Do not include all activities—choose only those that are essential and directly relevant. Distribute exactly {total_hours} hours among the selected activities.\n"
                f"Output only a CSV table with a header row: Category,Activity,Hours\nDo not output any explanation, extra text, or blank lines before or after the table. Do not use any other column names. Do not use '...' or blanks. If you cannot estimate hours for an activity, omit that activity from the table.\n\nExample:\nCategory,Activity,Hours\nDevelopment,Code development,10\nQA,Test cases,5\n\nThe sum of Hours must be exactly {total_hours}.\n\nActivities and Categories:\n{csv_activities}"
                + (f"\nInstructions: {instructions}" if instructions else "")
            )

        def build_adjustment_prompt(table_str):
            return (
                f"Please adjust the hours in the table below so that the sum of the Hours column is exactly {total_hours}. Output only a CSV table with a header row: Category,Activity,Hours.\nDo not output any explanation, extra text, or blank lines before or after the table.\n\n{table_str}"
            )

        if 'effort_attempts' not in st.session_state:
            st.session_state['effort_attempts'] = []
        if 'effort_manual_edit' not in st.session_state:
            st.session_state['effort_manual_edit'] = ''
        if 'effort_manual_mode' not in st.session_state:
            st.session_state['effort_manual_mode'] = False

        preview_prompt = build_effort_prompt()
        with st.expander("Preview/Edit Prompt", expanded=False):
            edited_prompt = st.text_area("Prompt to send", preview_prompt, key="edit_prompt_effort")
            use_edited = st.checkbox("Use edited prompt (overrides template)", key="use_edited_effort")

        def show_attempt(df_result, response, attempt_num):
            df = df_result['df'] if isinstance(df_result, dict) else df_result
            debug_csv = df_result.get('debug_csv') if isinstance(df_result, dict) else None
            header_cols = df_result.get('header_cols') if isinstance(df_result, dict) else None
            # Post-process: clamp, round, adjust
            if df is not None and 'Hours' in df.columns:
                df, changed, warn = clamp_and_adjust_hours(df, int(total_hours))
                if changed:
                    st.warning('Hours were clamped/adjusted to fit the total and range.')
                if warn:
                    st.error(warn)
            hours_sum = sum_hours_column(df) if df is not None else None
            st.markdown(f"<div class='effort-section'>Attempt {attempt_num}:</div>", unsafe_allow_html=True)
            if df is not None:
                st.dataframe(df, height=180)
                if hours_sum is not None:
                    st.info(f"Sum of Hours: {hours_sum}")
                else:
                    st.warning("Could not find a valid 'Hours' column or parse its values. Please check the table format.")
                output = BytesIO()
                df.to_excel(output, index=False)
                unique_key = f"download_excel_{attempt_num}_{uuid.uuid4()}"
                st.download_button("⬇️ Download Results as Excel", output.getvalue(), file_name="effort_allocation.xlsx", use_container_width=True, key=unique_key)
                # Add expander for raw LLM output for transparency/debugging
                with st.expander("Show raw LLM output", expanded=False):
                    st.code(response, language=None)
            else:
                st.error("Could not parse a table from the LLM response.")
                with st.expander("Show raw LLM output", expanded=True):
                    st.code(response, language=None)
                if header_cols is not None and not has_required_columns(header_cols):
                    st.error(f"Header row is missing required columns. Detected columns: {header_cols}. Required: Category, Activity, Hours.")
                    st.info("Tip: The table must have a header row with columns: Category, Activity, Hours (case-insensitive, in any order). Please fix in the manual edit area below.")

        def manual_edit_section():
            st.warning("All retries exhausted. You can manually edit the table below. Paste a markdown or CSV table with columns: Category, Activity, Hours.")
            st.session_state['effort_manual_edit'] = st.text_area("Manual Table Edit", st.session_state['effort_manual_edit'])
            if st.button("Validate Manual Table"):
                manual_df_result = extract_table_from_response(st.session_state['effort_manual_edit'])
                manual_df = manual_df_result['df'] if isinstance(manual_df_result, dict) else manual_df_result
                manual_sum = sum_hours_column(manual_df) if manual_df is not None else None
                if manual_df is not None and manual_sum == int(total_hours):
                    st.success("Manual table is valid! You can download it below.")
                    output = BytesIO()
                    manual_df.to_excel(output, index=False)
                    st.download_button("Download Results as Excel", output.getvalue(), file_name="effort_allocation.xlsx")
                elif manual_df is not None:
                    st.error(f"Sum of Hours is {manual_sum}, which does not match the requested total of {int(total_hours)}.")
                    st.dataframe(manual_df, height=180)
                else:
                    st.error("Could not parse a table from your manual input.")

        st.markdown('<div class="effort-section">4️⃣ Run Effort Estimation</div>', unsafe_allow_html=True)
        if st.button("🚀 Estimate Effort Allocation", use_container_width=True):
            st.session_state['effort_attempts'] = []
            st.session_state['effort_manual_mode'] = False
            prompt = edited_prompt if use_edited and edited_prompt else build_effort_prompt()
            with st.spinner("Estimating and distributing hours..."):
                response = call_llm(provider_choice, model_choice, prompt)
            df1 = extract_table_from_response(response)
            st.session_state['effort_attempts'].append((df1, response))

        for idx, (df_attempt, resp_attempt) in enumerate(st.session_state['effort_attempts']):
            show_attempt(df_attempt, resp_attempt, idx + 1)

        if st.session_state['effort_attempts']:
            last_df_result, last_resp = st.session_state['effort_attempts'][-1]
            last_df = last_df_result['df'] if isinstance(last_df_result, dict) else last_df_result
            last_sum = sum_hours_column(last_df) if last_df is not None else None
            if last_df is not None and last_sum == int(total_hours):
                st.success("Effort allocation complete! Total hours match.")
                st.markdown(f"<div style='background:#DCFCE7;padding:0.7em 1em;border-radius:8px;margin-bottom:1em;'><b>Summary:</b> {len(last_df)} activities, {last_sum} total hours</div>", unsafe_allow_html=True)
            elif len(st.session_state['effort_attempts']) < max_retries and not st.session_state['effort_manual_mode']:
                if st.button(f"🔄 Retry Adjustment (Attempt {len(st.session_state['effort_attempts']) + 1} of {max_retries})", use_container_width=True):
                    last_df = last_df_result['df'] if isinstance(last_df_result, dict) else last_df_result
                    table_str = last_df.to_markdown(index=False) if last_df is not None else last_resp
                    adjust_prompt = build_adjustment_prompt(table_str)
                    with st.spinner("Requesting adjustment from LLM..."):
                        adjust_response = call_llm(provider_choice, model_choice, adjust_prompt)
                    adjust_df = extract_table_from_response(adjust_response)
                    st.session_state['effort_attempts'].append((adjust_df, adjust_response))
            elif len(st.session_state['effort_attempts']) >= max_retries or st.session_state['effort_manual_mode']:
                st.session_state['effort_manual_mode'] = True
                manual_edit_section()

    st.markdown('<div class="effort-footer">Need help? <a href="https://streamlit.io" target="_blank">Learn more about Streamlit</a></div>', unsafe_allow_html=True) 