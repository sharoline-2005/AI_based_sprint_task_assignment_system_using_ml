"""
Sprint Planning Dashboard — talks to the FastAPI backend (app/main.py).

Run:
    streamlit run dashboard.py
(with the backend already running on http://localhost:8000)
"""
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"
SKILL_VOCAB = [
    "python", "java", "javascript", "react", "node", "sql", "aws",
    "docker", "kubernetes", "testing", "ui_ux", "devops",
    "machine_learning", "android", "ios", "spring_boot", "django",
]

st.set_page_config(page_title="AI Sprint Planner", layout="wide")
st.title("AI-Assisted Sprint Planning")
st.caption("Employee skill/experience analysis -> ML fit prediction -> optimized task assignment")

if "data_source" not in st.session_state:
    st.session_state.data_source = "synthetic"
if "jira_boards" not in st.session_state:
    st.session_state.jira_boards = None
if "jira_sprints" not in st.session_state:
    st.session_state.jira_sprints = None
if "jira_import" not in st.session_state:
    st.session_state.jira_import = None

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Data Source")
    st.session_state.data_source = st.radio(
        "Where should team/backlog data come from?",
        ["synthetic", "jira"],
        format_func=lambda v: "Synthetic demo data" if v == "synthetic" else "Import from Jira",
        index=0 if st.session_state.data_source == "synthetic" else 1,
    )

    if st.session_state.data_source == "jira":
        st.subheader("Connect Jira")
        site_url = st.text_input("Jira site URL", placeholder="https://yourteam.atlassian.net")
        email = st.text_input("Jira account email")
        api_token = st.text_input("API token", type="password",
                                   help="Create one at id.atlassian.com/manage-profile/security/api-tokens")

        if st.button("Connect & list boards"):
            if not (site_url and email and api_token):
                st.error("Fill in site URL, email, and API token first.")
            else:
                st.session_state.jira_creds = {
                    "site_url": site_url, "email": email, "api_token": api_token
                }
                try:
                    r = requests.post(f"{API_BASE}/jira/boards", json=st.session_state.jira_creds, timeout=20)
                    r.raise_for_status()
                    st.session_state.jira_boards = r.json()
                    st.session_state.jira_sprints = None
                    st.session_state.jira_import = None
                except requests.exceptions.HTTPError as e:
                    st.error(f"Connection failed: {e.response.json().get('detail', str(e))}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach backend/Jira: {e}")

        if st.session_state.jira_boards:
            board_options = {b["name"]: b["id"] for b in st.session_state.jira_boards}
            board_name = st.selectbox("Board", list(board_options.keys()))
            if st.button("Load sprints"):
                payload = {**st.session_state.jira_creds, "board_id": board_options[board_name]}
                r = requests.post(f"{API_BASE}/jira/sprints", json=payload, timeout=20)
                r.raise_for_status()
                st.session_state.jira_sprints = r.json()
                st.session_state.jira_board_id = board_options[board_name]

        if st.session_state.jira_sprints:
            sprint_options = {f'{s["name"]} ({s["state"]})': s["id"] for s in st.session_state.jira_sprints}
            sprint_name = st.selectbox("Sprint", list(sprint_options.keys()))
            if st.button("Import sprint from Jira", type="primary"):
                payload = {
                    **st.session_state.jira_creds,
                    "board_id": st.session_state.jira_board_id,
                    "sprint_id": sprint_options[sprint_name],
                }
                with st.spinner("Pulling issues and assignable users from Jira..."):
                    r = requests.post(f"{API_BASE}/jira/import", json=payload, timeout=30)
                    r.raise_for_status()
                    st.session_state.jira_import = r.json()
                st.success(
                    f"Imported {len(st.session_state.jira_import['tasks'])} tasks and "
                    f"{len(st.session_state.jira_import['employees'])} people."
                )


def build_synthetic_data():
    r_emp = requests.get(f"{API_BASE}/employees", timeout=10)
    r_emp.raise_for_status()
    r_task = requests.get(f"{API_BASE}/tasks", timeout=10)
    r_task.raise_for_status()
    employees_df = pd.DataFrame(r_emp.json())
    tasks_df = pd.DataFrame(r_task.json())
    employees_payload = employees_df.to_dict("records")
    tasks_payload = tasks_df.to_dict("records")
    return employees_df, tasks_df, employees_payload, tasks_payload


def build_jira_data():
    """
    Jira has no notion of skills/experience/capacity, so the imported team
    is shown as an editable table — the user fills these in once per sprint
    before running the AI planner.
    """
    imported = st.session_state.jira_import
    emp_df = pd.DataFrame(imported["employees"])
    if emp_df.empty:
        st.warning("No assignable users found on this Jira project.")
        return None, None, None, None

    # pre-fill from previously saved profiles (DB), so returning employees
    # don't need their skills/experience re-entered every sprint
    try:
        saved = requests.get(f"{API_BASE}/employees/profiles", timeout=10).json()
    except requests.exceptions.RequestException:
        saved = {}

    def _prefill(col, default):
        return emp_df["employee_id"].map(lambda eid: saved.get(eid, {}).get(col, default))

    emp_df["role"] = _prefill("role", "Team Member")
    emp_df["experience_years"] = _prefill("experience_years", 2.0)
    emp_df["velocity_avg"] = _prefill("velocity_avg", 8.0)
    emp_df["capacity_points"] = _prefill("capacity_points", 10.0)
    emp_df["skills_csv"] = emp_df["employee_id"].map(
        lambda eid: ", ".join(saved.get(eid, {}).get("skills", {}).keys())
    )

    st.info(
        "Jira doesn't track skills/experience — fill these in below (pre-filled from previously "
        f"saved profiles where available). Valid skill keywords: {', '.join(SKILL_VOCAB)}"
    )
    edited = st.data_editor(
        emp_df[["employee_id", "name", "role", "experience_years",
                "velocity_avg", "capacity_points", "skills_csv"]],
        column_config={
            "skills_csv": st.column_config.TextColumn(
                "skills (comma-separated)", help="e.g. python, react, sql"
            ),
        },
        use_container_width=True, hide_index=True, key="employee_editor",
    )

    employees_payload = []
    for _, row in edited.iterrows():
        skills = {s.strip().lower(): 0.8 for s in str(row["skills_csv"]).split(",") if s.strip()}
        employees_payload.append({
            "employee_id": row["employee_id"], "name": row["name"], "role": row["role"],
            "experience_years": float(row["experience_years"]), "skills": skills,
            "velocity_avg": float(row["velocity_avg"]), "capacity_points": float(row["capacity_points"]),
            "current_load_points": 0.0,
        })
    employees_df = edited.drop(columns=["skills_csv"])

    if st.button("Save team profile (so next sprint pre-fills automatically)"):
        for emp in employees_payload:
            profile = {k: v for k, v in emp.items() if k != "current_load_points"}
            requests.post(f"{API_BASE}/employees/profile", json=profile, timeout=10)
        st.success(f"Saved {len(employees_payload)} employee profiles.")

    tasks_df = pd.DataFrame(imported["tasks"])
    tasks_df["story_points"] = tasks_df["story_points"].fillna(3).astype(float)
    tasks_payload = []
    for _, row in tasks_df.iterrows():
        tasks_payload.append({
            "task_id": row["task_id"], "sprint_id": "JIRA-IMPORT", "title": row["title"],
            "description": row["description"] or row["title"], "story_points": row["story_points"],
            "task_type": row["task_type"], "priority": int(row["priority"]),
        })
    return employees_df, tasks_df[["task_id", "title", "task_type", "story_points", "priority"]], \
        employees_payload, tasks_payload


# ------------------------------------------------------------- load data ---
employees_df = tasks_df = employees_payload = tasks_payload = None

if st.session_state.data_source == "synthetic":
    try:
        employees_df, tasks_df, employees_payload, tasks_payload = build_synthetic_data()
    except requests.exceptions.RequestException:
        st.error(
            "Can't reach the backend API at http://localhost:8000. "
            "Start it with `uvicorn app.main:app --reload` inside `backend/`."
        )
        st.stop()
elif st.session_state.jira_import:
    employees_df, tasks_df, employees_payload, tasks_payload = build_jira_data()
else:
    st.info("Connect to Jira and import a sprint from the sidebar to get started.")
    st.stop()

if employees_df is None or tasks_df is None or employees_df.empty or tasks_df.empty:
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Team ({len(employees_df)})")
    st.dataframe(employees_df, use_container_width=True, hide_index=True)
with col2:
    st.subheader(f"Sprint Backlog ({len(tasks_df)})")
    st.dataframe(tasks_df, use_container_width=True, hide_index=True)

st.divider()

if st.button("Run AI Sprint Planning", type="primary"):
    with st.spinner("Predicting fit scores and optimizing assignments..."):
        body = {"employees": employees_payload, "tasks": tasks_payload}
        r = requests.post(f"{API_BASE}/sprint/plan", json=body, timeout=30)
        r.raise_for_status()
        result = r.json()

    # stashed in session_state so the outcome-logging form below survives
    # the rerun triggered by its own widgets/button
    st.session_state.last_plan = {
        "result": result,
        "employees_df": employees_df,
        "tasks_df": tasks_df,
    }

plan = st.session_state.get("last_plan")
if plan and set(plan["employees_df"]["employee_id"]) == set(employees_df["employee_id"]) \
        and set(plan["tasks_df"]["task_id"]) == set(tasks_df["task_id"]):
    result = plan["result"]
    assignments = pd.DataFrame(result["assignments"])
    unassigned = result["unassigned_task_ids"]

    if assignments.empty:
        st.warning("No assignments could be made.")
    else:
        merged = assignments.merge(
            employees_df[["employee_id", "name", "role"]], on="employee_id"
        ).merge(
            tasks_df[["task_id", "title", "story_points", "task_type"]], on="task_id"
        )

        st.subheader("Recommended Assignments")
        st.dataframe(
            merged[["task_id", "title", "employee_id", "name", "role",
                    "story_points", "fit_score", "predicted_days", "reason"]]
            .sort_values("fit_score", ascending=False),
            use_container_width=True, hide_index=True,
        )

        if unassigned:
            st.warning(f"{len(unassigned)} task(s) could not be assigned "
                       f"(team at capacity): {', '.join(unassigned)}")

        st.subheader("Workload Balance After Assignment")
        assigned_points = assignments.merge(
            tasks_df[["task_id", "story_points"]], on="task_id"
        ).groupby("employee_id")["story_points"].sum().rename("assigned_points")

        workload = employees_df[["employee_id", "name", "capacity_points"]].merge(
            assigned_points, on="employee_id", how="left"
        )
        workload["assigned_points"] = workload["assigned_points"].fillna(0)
        workload["utilization_%"] = (
            workload["assigned_points"] / workload["capacity_points"] * 100
        ).round(0)

        fig = px.bar(
            workload, x="name", y="utilization_%",
            title="Capacity Utilization per Employee",
            labels={"utilization_%": "Utilization (%)", "name": "Employee"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Fit Score by Task Priority")
        merged_pri = merged.merge(tasks_df[["task_id", "priority"]], on="task_id")
        fig2 = px.bar(
            merged_pri.groupby("priority")["fit_score"].mean().reset_index(),
            x="priority", y="fit_score", title="Assignment Quality by Priority",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Log Actual Outcomes (feedback loop)")
        st.caption(
            "Once the sprint actually runs, record what really happened here. "
            "These get stored in the database and can be used to retrain the model on real data."
        )
        outcome_df = merged[["task_id", "employee_id", "title"]].copy()
        outcome_df["actual_success"] = True
        outcome_df["actual_days"] = merged["predicted_days"]
        edited_outcomes = st.data_editor(
            outcome_df, use_container_width=True, hide_index=True, key="outcome_editor",
            column_config={
                "actual_success": st.column_config.CheckboxColumn("Completed successfully?"),
                "actual_days": st.column_config.NumberColumn("Actual days taken", min_value=0.0),
            },
        )
        if st.button("Save outcomes to database"):
            for _, row in edited_outcomes.iterrows():
                task_row = merged[merged["task_id"] == row["task_id"]].iloc[0]
                requests.post(f"{API_BASE}/sprint/outcomes", json={
                    "sprint_id": tasks_df.iloc[0].get("sprint_id", "JIRA-IMPORT") if "sprint_id" in tasks_df.columns else "SPRINT-CURRENT",
                    "task_id": row["task_id"], "employee_id": row["employee_id"],
                    "task_type": str(task_row["task_type"]), "story_points": float(task_row["story_points"]),
                    "predicted_fit_score": float(task_row["fit_score"]),
                    "predicted_days": float(task_row["predicted_days"]),
                    "actual_success": bool(row["actual_success"]),
                    "actual_days": float(row["actual_days"]),
                }, timeout=10)
            st.success(f"Logged {len(edited_outcomes)} outcomes.")
