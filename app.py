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
    ("Front Wing Mainplane Layup", 12),
    ("Vacuum Bagging", 2),
    ("Autoclave Cure", 8),
    ("Precision Trim & Drill", 4),
    ("Chassis Assembly", 6),
    ("Final Inspection & QA", 3),
]

# Define realistic F1 scenarios
SCENARIOS = [
    {
        "name": "Baseline (No Issues)",
        "description": "All systems normal. No delays or breakdowns.",
        "vars": {"delay": 0, "workers": 5, "autoclave_capacity": 1, "overtime_hours": 0, "shift_start_hour": 8}
    },
    {
        "name": "Carbon Fiber Delivery Delay",
        "description": "A 12-hour delay in carbon fiber delivery for the front wing mainplane.",
        "vars": {"delay": 12}
    },
    {
        "name": "Autoclave Breakdown",
        "description": "Autoclave capacity reduced to 0.5 (half speed) for 8 hours.",
        "vars": {"autoclave_capacity": 0.5}
    },
    {
        "name": "Sudden Regulatory Change",
        "description": "Inspection time increased by 2 hours due to new FIA regulation.",
        "vars": {"delay": 0, "workers": 5, "autoclave_capacity": 1, "overtime_hours": 0, "shift_start_hour": 8},
        "task_mod": {"Final Inspection & QA": 5}
    },
    {
        "name": "Overtime Restrictions",
        "description": "No overtime allowed due to labor law changes.",
        "vars": {"overtime_hours": 0}
    },
    {
        "name": "Last-Minute Design Change",
        "description": "Trim & Drill task duration increased by 2 hours for urgent design update.",
        "vars": {},
        "task_mod": {"Precision Trim & Drill": 6}
    },
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
    # Determine which tasks are delayed (start time > baseline start)
    baseline_start = df["Start"].min()
    df = df.copy()
    df["Status"] = "Normal"
    # Mark tasks as 'Delayed' if their start is after the baseline
    df.loc[df["Start"] > baseline_start, "Status"] = "Delayed"
    df.loc[df["Task"].str.contains("Autoclave", case=False), "Status"] = "Autoclave"
    color_scale = alt.Scale(domain=["Normal", "Autoclave", "Delayed"], range=["#1b263b", "#f4a259", "#d90429"])
    chart = alt.Chart(df).mark_bar().encode(
        x='Start',
        x2='End',
        y=alt.Y('Task', sort=None),
        color=alt.Color('Status', scale=color_scale, legend=alt.Legend(
            title="Task Status",
            orient="bottom",
            labelFontSize=13,
            titleFontSize=14,
            symbolSize=200,
            labelColor="#222"
        )),
        tooltip=['Task', 'Start', 'End', 'Duration', 'Status']
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
    st.markdown("## 🏁 Pit Wall Chat")
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
                f"You are an F1 pit wall operations strategist. A user has made the following scenario change(s) in a composite part build schedule: {user_instruction}\n"
                f"Current scenario variables: delay={st.session_state['delay']}, shift_start_hour={st.session_state['shift_start_hour']}, "
                f"overtime_hours={st.session_state['overtime_hours']}, workers={st.session_state['workers']}, autoclave_capacity={st.session_state['autoclave_capacity']}.\n"
                "Reply in a conversational, expert tone as if you are advising the F1 operations team. Confirm the change, explain its likely impact, and proactively suggest at least one mitigation or next step the team should consider."
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=180,
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
    st.markdown(
        """
        <div style='display:flex;align-items:center;margin-bottom:8px;'>
            <span style='font-size:2.2em;margin-right:12px;'>🏁</span>
            <span style='font-size:2em;font-weight:700;color:#d90429;text-shadow:1px 1px 0 #fff;'>What-If Analysis Powered by AI</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='font-size:1.1em; background:linear-gradient(90deg,#f5f7fa 80%,#d90429 100%); border-radius:8px; padding:18px 20px; margin-bottom:18px;'>
        <b>Scenario:</b> It's Thursday before the <b>Monaco Grand Prix</b>. A key supplier notifies you of a delay in <b>carbon fiber delivery</b> for the front wing mainplane. The clock is ticking. What's the impact on your build schedule? How can your team adapt in real time to ensure you're ready for the race?
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "Interactively explore how a supplier delay ripples through the build schedule for a front-wing mainplane. Use the chat to make scenario changes and let the AI propose mitigations."
    )
    scenario_names = [s["name"] for s in SCENARIOS]
    selected_idx = st.selectbox("🏁 Choose a Scenario", scenario_names, index=0)
    selected_scenario = SCENARIOS[scenario_names.index(selected_idx)]
    if st.button("Apply Scenario", key="apply_scenario"):
        update_scenario_vars(selected_scenario.get("vars", {}))
        if "task_mod" in selected_scenario:
            for tname, tval in selected_scenario["task_mod"].items():
                for i, (name, hrs) in enumerate(TASKS):
                    if name == tname:
                        TASKS[i] = (name, tval)
        st.session_state["chat_history"].append({
            "role": "bot",
            "content": f"🏁 <b>Scenario Applied:</b> {selected_scenario['name']}<br>{selected_scenario['description']}"
        })
        st.rerun()
    if st.button("Reset to Baseline", key="reset_scenario"):
        update_scenario_vars(SCENARIOS[0]["vars"])
        st.session_state["chat_history"].append({
            "role": "bot",
            "content": "🔄 <b>Scenario reset to baseline.</b> All systems normal."
        })
        st.rerun()
    schedule_df = compute_schedule(
        TASKS,
        st.session_state["delay"],
        st.session_state["shift_start_hour"],
        st.session_state["overtime_hours"],
        st.session_state["workers"],
        st.session_state["autoclave_capacity"]
    )
    st.subheader("🏎️ Race Readiness Timeline (Gantt Chart)")
    st.altair_chart(gantt_chart(schedule_df), use_container_width=True)
    st.markdown(
        """
        <div style='margin-bottom:16px;'><b>Legend:</b> <span style='color:#1b263b;font-weight:600;'>■ Normal</span> <span style='color:#f4a259;font-weight:600;'>■ Autoclave</span> <span style='color:#d90429;font-weight:600;'>■ Delayed/Critical</span></div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(schedule_df)
    st.subheader("🧑‍💼 Management Summary (AI-Powered)")
    summary = openai_management_summary(schedule_df, {
        "delay": st.session_state["delay"],
        "shift_start_hour": st.session_state["shift_start_hour"],
        "overtime_hours": st.session_state["overtime_hours"],
        "workers": st.session_state["workers"],
        "autoclave_capacity": st.session_state["autoclave_capacity"],
    })
    st.write(summary)
