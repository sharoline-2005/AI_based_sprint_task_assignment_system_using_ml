# Sharoline ML Project — AI-Based Sprint Task Assignment System

Analyzes employee technical skills and experience, and recommends the optimal
task assignment for each sprint using an ML prediction model + optimization
(Hungarian Algorithm) layer. Can run on synthetic demo data or import a real
sprint directly from Jira.




## Architecture

```
backend/
  app/
    models/         # Pydantic schemas (Employee, Task, Assignment, Jira DTOs)
    ml/             # Skill extraction (NLP), prediction model, optimizer
    api/            # FastAPI routes (sprint planning + Jira integration)
    data/           # Synthetic dataset generator
    integrations/   # Jira Cloud REST API client
    db/             # SQLAlchemy models + persistence (employee profiles, outcomes)
  data/             # Generated CSV files + app.db (SQLite)
frontend/
  dashboard.py      # Streamlit sprint planning dashboard
```

## Pipeline

1. **Data Ingestion** — synthetic generator, or live import from a Jira board/sprint
2. **Feature Engineering** — skill vectors, experience scores, task skill extraction (keyword/NLP)
3. **Prediction Engine** — RandomForest predicts fit-score (success probability) + expected completion days per employee-task pair
4. **Optimization Engine** — Hungarian Algorithm assigns tasks across the whole team, maximizing total fit while respecting exact story-point capacity
5. **Dashboard** — Streamlit UI to run sprint planning, view AI-recommended assignments, and log real outcomes back into the database
6. **Persistence** — employee skill/experience profiles and logged sprint outcomes are saved to a database (SQLite by default, Postgres-ready)

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m app.data.generate_synthetic_data   # creates data/employees.csv, tasks.csv, history.csv
python -m app.ml.train_model                 # trains and saves the fit-prediction model
uvicorn app.main:app --reload                # starts API on http://localhost:8000
```

In a second terminal:

```bash
cd frontend
pip install -r requirements.txt
streamlit run dashboard.py
```

Open `http://localhost:8501`. In the sidebar, choose **Synthetic demo data**
(works immediately) or **Import from Jira**.

## Connecting a real Jira board

1. Create a free Jira Cloud site (atlassian.com) with a Scrum board and a
   sprint with a few issues.
2. Generate an API token: id.atlassian.com/manage-profile/security/api-tokens
3. In the dashboard sidebar: pick **Import from Jira**, enter your site URL,
   email, and API token, connect, pick a board and sprint, then **Import**.
4. Jira has no concept of skills/experience, so the imported team appears in
   an editable table — fill those in once. Click **Save team profile** to
   persist them (auto-filled on future imports for the same people).

## Database / persistence

Defaults to a local SQLite file (`backend/data/app.db`) so the project runs
with zero extra infrastructure. The exact same SQLAlchemy models work
unchanged against a real Postgres server — just set an environment variable
before starting the API:

```bash
set DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/sprint_ai
uvicorn app.main:app --reload
```

Two tables:
- `employee_profiles` — saved skill/experience/capacity per person
- `assignment_outcomes` — actual results logged after a sprint runs (the
  feedback loop: point `app/ml/train_model.py` at this table instead of the
  synthetic `history.csv` once you have enough real data to retrain on)

## Next Steps
- Retrain the fit model periodically on `assignment_outcomes` instead of synthetic data
- Add authentication (the Jira credentials + DB are currently open on localhost)
- Deploy with Docker Compose (FastAPI + Streamlit + Postgres) for a shareable demo
