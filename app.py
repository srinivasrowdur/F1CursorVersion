"""
Streamlit demo app – Advanced "What-If" analysis for a composite part with live ChatGPT feedback
Scenario: *Supplier delay – carbon-cloth shipment slips by up to 72 h*
"""

import os
import streamlit as st
import pandas as pd
import altair as alt
import openai
import datetime as dt
import re
from dotenv import load_dotenv
from typing import List

# --- Baseline task data (hrs) ---
TASKS: List[tuple[str, int]] = [
    ("Lay-up", 12),
    ("Vacuum-bag", 2),
    ("Autoclave", 8),
    ("Trim / Drill", 4),
    ("Assembly", 6),
    ("Inspection", 3),
]

# --- Scenario Variables: initialized and managed via chat ---
def get_default(var, default):
    if var in st.session_state:
        return st.session_state[var]
    st.session_state[var] = default
    return default

def update_scenario_vars(vars_dict):
    for k, v in vars_dict.items():
        st.session_state[k] = v

def parse_and_apply_nl_instruction(text):
    changes = {}
    patterns = {
        "shift_start_hour": r"shift start.*?(\d{1,2})",
        "overtime_hours": r"overtime.*?(\d{1,2})",
        "workers": r"workers?\D*(\d{1,2})",
        "autoclave_capacity": r"autoclave.*?capacit.*?(\d{1,2})",
        "delay": r"delay.*?(\d{1,3})",
    }
    for var, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            changes[var] = val
    if changes:
        update_scenario_vars(changes)
    return changes

def compute_schedule(tasks, delay, shift_start_hour, overtime_hours, workers, autoclave_capacity):
    schedule = []
    start = dt.datetime(2025, 5, 21, shift_start_hour, 0)
    worker_factor = max(1, workers / 5)
    for i, (name, hrs) in enumerate(tasks):
        adj_hrs = max(1, int(hrs / worker_factor)) if name != "Autoclave" else hrs
        if i == 1:
            start += dt.timedelta(hours=delay)
        if overtime_hours > 0:
            adj_hrs = max(1, adj_hrs - overtime_hours // len(tasks))
        if name == "Autoclave" and autoclave_capacity > 1:
            adj_hrs = max(1, int(adj_hrs / autoclave_capacity))
        end = start + dt.timedelta(hours=adj_hrs)
        schedule.append({
            "Task": name,
            "Start": start,
            "End": end,
            "Duration": adj_hrs,
        })
        start = end
    return pd.DataFrame(schedule)

def gantt_chart(df):
    chart = alt.Chart(df).mark_bar().encode(
        x='Start',
        x2='End',
        y=alt.Y('Task', sort=None),
        color=alt.condition(
            alt.datum.Task == 'Autoclave',
            alt.value('orange'),
            alt.value('steelblue')
        ),
        tooltip=['Task', 'Start', 'End', 'Duration']
    ).properties(height=300)
    return chart

def openai_management_summary(schedule_df, scenario_vars):
    prompt = (
        f"A composite part build schedule has the following tasks and durations (in hours):\n"
        f"{schedule_df[['Task','Duration']].to_dict(orient='records')}\n"
        f"Scenario variables: {scenario_vars}\n"
        "Provide a concise management summary of the impact and suggest 3 mitigation ideas."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI Error] {e}"

def describe_scenario_change(changes):
    descriptions = []
    for k, v in changes.items():
        if k == "delay":
            descriptions.append(f"Supplier delay set to {v} hours. This may push the schedule later and impact downstream tasks.")
        elif k == "workers":
            descriptions.append(f"Number of workers set to {v}. More workers can speed up manual tasks, fewer will slow them down.")
        elif k == "shift_start_hour":
            descriptions.append(f"Shift start hour set to {v}:00. This changes when the workday begins.")
        elif k == "overtime_hours":
            descriptions.append(f"Overtime set to {v} hours per day. This can reduce total build time.")
        elif k == "autoclave_capacity":
            descriptions.append(f"Autoclave capacity set to {v}. Higher capacity can reduce the autoclave bottleneck.")
        else:
            descriptions.append(f"{k} set to {v}.")
    return " ".join(descriptions)

# --- Streamlit page config ---
st.set_page_config(page_title="Composite What-If Demo", layout="wide")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    import warnings
    warnings.warn("OPENAI_API_KEY not found in environment. Please set it in your .env or environment variables.")

get_default("shift_start_hour", 8)
get_default("overtime_hours", 0)
get_default("workers", 5)
get_default("autoclave_capacity", 1)
get_default("delay", 0)

# --- Two-column layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## 💬 Scenario Chat")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(f"<div style='background:#e6f0ff;padding:8px 12px;border-radius:8px;margin-bottom:4px;text-align:right'><b>You:</b> {entry['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#f4f4f4;padding:8px 12px;border-radius:8px;margin-bottom:4px;text-align:left'><b>Bot:</b> {entry['content']}</div>", unsafe_allow_html=True)
    user_instruction = st.text_input("Type your scenario change (e.g., 'Set delay to 24 hours')", key="chat_input")
    send = st.button("Send", key="send_button")
    if send and user_instruction:
        st.session_state["chat_history"].append({"role": "user", "content": user_instruction})
        changes = parse_and_apply_nl_instruction(user_instruction)
        if changes:
            prompt = (
                f"A user has made the following scenario change(s) in a composite part build schedule: {user_instruction}\n"
                f"Current scenario variables: delay={st.session_state['delay']}, shift_start_hour={st.session_state['shift_start_hour']}, "
                f"overtime_hours={st.session_state['overtime_hours']}, workers={st.session_state['workers']}, autoclave_capacity={st.session_state['autoclave_capacity']}.\n"
                "Reply with a friendly confirmation of the change and a brief, clear explanation of its likely impact on the build schedule."
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=150,
                    stream=False,
                )
                print(f"Full OpenAI API response: {response}")
                bot_msg = response.choices[0].message.content.strip()
                if not bot_msg:
                    bot_msg = "(No response from AI. Please try again or check your API settings.)"
            except Exception as e:
                bot_msg = f"[OpenAI Error] {e}"
        else:
            bot_msg = "No changes detected. Please try a different instruction."
        st.session_state["chat_history"].append({"role": "bot", "content": bot_msg})
        print(f"Bot response: {bot_msg}")
        st.rerun()

with col2:
    st.title("🤖 What-If Analysis Powered by AI")
    st.markdown(
        "Interactively explore how a supplier delay ripples through the build schedule for a front-wing mainplane. Use the chat to make scenario changes and let the AI propose mitigations."
    )
    schedule_df = compute_schedule(
        TASKS,
        st.session_state["delay"],
        st.session_state["shift_start_hour"],
        st.session_state["overtime_hours"],
        st.session_state["workers"],
        st.session_state["autoclave_capacity"]
    )
    st.subheader("Build Schedule Gantt Chart")
    st.altair_chart(gantt_chart(schedule_df), use_container_width=True)
    st.dataframe(schedule_df)
    st.subheader("Management Summary (AI-Powered)")
    summary = openai_management_summary(schedule_df, {
        "delay": st.session_state["delay"],
        "shift_start_hour": st.session_state["shift_start_hour"],
        "overtime_hours": st.session_state["overtime_hours"],
        "workers": st.session_state["workers"],
        "autoclave_capacity": st.session_state["autoclave_capacity"],
    })
    st.write(summary)
