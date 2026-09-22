# Sharoline ML Project — End-to-End Documentation

**AI-Based Sprint Task Assignment System**
Analyzes employee skills/experience and recommends the optimal task assignment for each sprint using an ML prediction model + optimization (Hungarian Algorithm) layer.

---

## 1. High-Level Architecture

```
backend/                       FastAPI service (port 8000)
  app/
    models/schema.py           Pydantic request/response contracts + skill taxonomy
    ml/                        NLP skill extraction, ML fit prediction, Hungarian optimizer, training script
    api/                       REST routes (core + Jira)
    db/                        SQLAlchemy models/session for persistence
    integrations/              Jira Cloud REST API client
    data/                      Synthetic dataset generator
  data/                        Generated CSVs, trained model files, SQLite DB
  requirements.txt

frontend/
  dashboard.py                 Streamlit UI (port 8501) — talks to backend over HTTP
  requirements.txt

.claude/launch.json             Dev launch config for backend + frontend
```

**Why this split:** Backend and frontend are two fully decoupled processes communicating only over HTTP (`http://localhost:8000/api`). This means the ML/optimization logic is reusable by any client (CLI, another UI, a cron job) — not locked into Streamlit.

---

## 2. Pipeline (Why each stage exists)

1. **Data Ingestion** — either synthetic demo data (CSV) or a live import from Jira (via `JiraClient`). Two sources feed the *same* schema so the rest of the pipeline doesn't care where data came from.
2. **Feature Engineering** — turn raw skills/task text into numeric features: skill vectors (proficiency 0–1 per skill), a weighted `skill_match` score between task requirements and employee skills, experience, velocity, current workload.
3. **Prediction Engine** — a RandomForest model predicts, for every (employee, task) pair, the probability of successful completion and the expected number of days.
4. **Optimization Engine** — the Hungarian Algorithm (via `scipy.optimize.linear_sum_assignment`) finds the assignment that maximizes total fit across the *whole team* simultaneously (not greedily per-task), while respecting each employee's remaining story-point capacity.
5. **Dashboard** — a Streamlit UI to trigger planning, inspect the plan, and feed real outcomes back in for future retraining.

---

## 3. Backend — File-by-File

### 3.1 `backend/app/models/schema.py` — the domain contract
All Pydantic models live in one file, plus two shared constants:
- `SKILLS` — fixed vocabulary of 17 canonical skills (python, java, javascript, react, node, sql, aws, docker, kubernetes, testing, ui_ux, devops, machine_learning, android, ios, spring_boot, django). Every skill vector (employee or task) is scored against this fixed list — **why fixed**: keeps the ML feature space stable and comparable across employees/tasks/sprints.
- `TASK_TYPES` — `feature`, `bug`, `testing`, `devops`, `research`.
- **`Employee`**: id, name, role, experience_years, `skills: Dict[str, float]` (proficiency 0–1), velocity_avg (avg story points/sprint), capacity_points, current_load_points.
- **`Task`**: id, sprint_id, title, description, story_points, task_type, priority (1=highest…5=lowest), optional `required_skills` (auto-filled by the skill extractor if missing — this is what lets Jira-imported tasks, which have no skill tags, still work).
- **`Assignment`**: task_id, employee_id, fit_score, predicted_days, human-readable `reason`.
- **`SprintPlanRequest` / `SprintPlanResponse`**: input (employees + tasks) and output (assignments + `unassigned_task_ids`) of the planning endpoint.
- **`JiraCredentials`**, **`JiraBoardsRequest`**, **`JiraSprintsRequest`**, **`JiraImportRequest`**, **`ImportedEmployee`**, **`ImportedTask`**, **`JiraImportResponse`** — Jira integration contracts.
- **`EmployeeProfileIn`** — persisted-profile input with defaults (role="Team Member", experience=2.0, velocity=8.0, capacity=10.0) so a partially-filled form still validates.
- **`OutcomeIn`** — feedback-loop input: predicted vs. actual fit/days/success per assignment, for future retraining.

### 3.2 `backend/app/ml/` — the ML pipeline

**`skill_extractor.py`** — keyword/synonym matcher, *not* a trained NLP model.
Why keyword matching instead of a trained classifier: task descriptions are short and skill names are a closed, known vocabulary, so a synonym-normalized regex match (`js`→`javascript`, `k8s`→`kubernetes`, `postgres`/`mysql`→`sql`, `qa`→`testing`, `ux`/`ui`→`ui_ux`, `ci/cd`→`devops`, etc.) is cheap, deterministic, and debuggable — no training data or model drift to manage for something this simple. Returns `{}` (not zero-penalty) when no skill is detected, so `skill_match()` treats an untagged task as *neutral* rather than unfairly penalizing every employee.

**Feature engineering — `skill_match()`** (defined in `generate_synthetic_data.py`, reused everywhere):
```python
def skill_match(employee_skills, required_skills) -> float:
    if not required_skills:
        return 0.5
    num, den = 0.0, 0.0
    for skill, weight in required_skills.items():
        den += weight
        num += weight * employee_skills.get(skill, 0.0)
    return round(num / den, 3) if den > 0 else 0.5
```
A weighted overlap score (0–1) — weights let a task say "React is critical, testing is nice-to-have" rather than treating every required skill equally.

**`predictor.py` — `FitPredictor`**
Loads two pre-trained `joblib` pipelines at construction (raises a clear `FileNotFoundError` telling the user to run `train_model` if they're missing). `score_pairs(employees, tasks)` builds every (employee × task) pair as a row with features `task_type`, `skill_match`, `experience_years`, `velocity_avg`, `workload_at_assignment`, `story_points`, then predicts:
- `fit_score` = `fit_model.predict_proba(X)[:, 1]` (probability of success)
- `predicted_days` = `days_model.predict(X)`

**`train_model.py` — training script**
- **Algorithm**: `RandomForestClassifier` (success probability) + `RandomForestRegressor` (completion days), both `n_estimators=200, max_depth=8, random_state=42`.
  **Why RandomForest**: robust to mixed categorical/numeric features, doesn't need feature scaling, resists overfitting on a modest synthetic dataset, and gives free feature-importance insight — a reasonable, low-maintenance default before reaching for gradient boosting or deep learning.
- **Preprocessing**: `ColumnTransformer` + `OneHotEncoder(categories=[TASK_TYPES], handle_unknown="ignore")` on `task_type`, wrapped in an sklearn `Pipeline` so the encoder and model are serialized together — **why**: guarantees inference-time preprocessing always exactly matches training-time preprocessing (a common source of silent bugs when encoder and model are saved/loaded separately).
- Trains on `data/history.csv` (80/20 split), reports ROC-AUC (classifier) and MAE (regressor), saves both pipelines with `joblib.dump`.
- Run with: `python -m app.ml.train_model`. The script's own docstring notes it can later train against the real `assignment_outcomes` DB table instead of synthetic history, once enough real feedback has been logged.

**`optimizer.py` — Hungarian Algorithm assignment**
**Why Hungarian Algorithm over greedy assignment**: greedily giving each task to whoever scores highest can starve other tasks of good-fit employees and produce a *worse total team outcome* even though each individual pick looked locally optimal. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) solves the assignment problem optimally in polynomial time — it finds the single best *global* pairing.

Two-stage design (because plain Hungarian assumes 1 task per "slot", but real capacity is in story points, not head-count):
- **Stage 1** (`_stage1_candidates`): each employee is expanded into `N = floor(remaining_capacity / avg_story_points)` capacity slots (min 1). Builds a cost matrix `cost = 1 - fit_score` over `(slots × tasks)` and solves it with `linear_sum_assignment` to get a globally-preferred employee per task, ignoring exact per-task point sizes (since slots are only average-sized).
- **Stage 2** (`optimize_assignment`): processes tasks **best-fit-first** (sorted by Stage-1 preferred employee's fit score, descending). Assigns the Stage-1-preferred employee if they still have enough exact story-point capacity left; otherwise re-routes to whichever *sufficiently-capacitated* employee has the next-best fit score. Capacity is decremented as tasks are assigned in order. Tasks nobody has room for land in `unassigned_task_ids`.
- Returns `List[Assignment]` with a human-readable `reason` (e.g. *"Predicted 82% success probability, ~3.4 days to complete"*).

### 3.3 `backend/app/api/` — REST routes

**`routes.py`** (mounted at `/api`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/employees` | Load synthetic employees from `data/employees.csv` |
| GET | `/api/tasks` | Load synthetic backlog tasks from `data/tasks.csv` |
| POST | `/api/sprint/plan` | Full pipeline: skill match → ML fit prediction → Hungarian optimization. Body is optional — omit it to fall back to the synthetic CSVs |
| POST | `/api/employees/profile` | Persist/update an employee's skill/experience profile to the DB |
| GET | `/api/employees/profiles` | Fetch all saved profiles (used to pre-fill the dashboard after a Jira import) |
| POST | `/api/sprint/outcomes` | Feedback loop: log a real sprint outcome |
| GET | `/api/sprint/outcomes` | List all logged outcomes |

**`jira_routes.py`** (mounted at `/api/jira`): `POST /boards`, `POST /sprints`, `POST /import` — credentials are passed in the request body per-call and never persisted server-side. `JiraAuthError` → HTTP 401, network errors → HTTP 502.

### 3.4 `backend/app/data/generate_synthetic_data.py`
Generates a full synthetic world so the app is runnable with zero external dependencies. Seeded (`random.seed(42)`, `np.random.seed(42)`) for reproducibility.
- **`employees.csv`** (25 employees): role-biased skill vectors (primary/role skills get proficiency 0.6–1.0, others 0.1–0.5 with 35% chance of appearing at all), experience/velocity/capacity drawn from realistic distributions (exponential/normal, clipped).
- **`tasks.csv`** (20 backlog tasks): Fibonacci-like story points, random task type/priority, 1–3 required skills sampled from a role's primary skill set.
- **`history.csv`** (3000 rows): the ML training set. Success is simulated via a logistic model (`logit = -1.5 + 3.2*skill_match + 0.12*min(experience,10) - 0.05*story_points + noise`) — **why this formula**: encodes the intuitive ground truth that skill match and experience should *increase* success odds while task size should *decrease* it, giving the RandomForest something real to learn. Completion days similarly scale with task size, inversely with skill match and velocity, plus noise.

### 3.5 `backend/app/db/` — Persistence
- **`database.py`**: SQLAlchemy engine/session, defaulting to `sqlite:///backend/data/app.db`; reads `DATABASE_URL` env var so swapping to Postgres in production is a one-line config change, not a code change (`psycopg2-binary` is already in requirements for this).
- **`models.py`**: `EmployeeProfile` (persist skill/experience data keyed by employee ID, since Jira has no concept of "skills" — this is what lets the dashboard remember a team's profile across Jira imports) and `AssignmentOutcome` (logs predicted vs. actual results — the raw material for eventually retraining on *real* data instead of synthetic history).
- **`crud.py`**: simple upsert/query helpers used by the routes.

### 3.6 `backend/app/integrations/jira_client.py` — `JiraClient`
Thin wrapper over Jira Cloud REST API v3 + Agile API 1.0, using HTTP Basic Auth with an email + API token (Atlassian's documented method for Cloud). Handles pagination, looks up the "Story Points" custom field dynamically (Jira doesn't have a fixed field ID for it — every instance can differ), flattens Jira's rich-text description format (ADF) into plain text for the skill extractor, and maps Jira priority names/issue types into this app's own vocabulary.

### 3.7 `backend/app/main.py` — Wiring
```python
app = FastAPI(title="Sprint Assignment AI", ...)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")
app.include_router(jira_router, prefix="/api/jira", tags=["jira"])

@app.on_event("startup")
def on_startup():
    init_db()
```
CORS is wide open (`*`) — fine for local development, flagged in the README as something to lock down (plus add authentication) before any real deployment.

---

## 4. Frontend — `frontend/dashboard.py`

Single-file Streamlit app. `API_BASE = "http://localhost:8000/api"` — the frontend never imports backend code directly; it's a pure HTTP client, keeping the two deployable independently.

**User workflow:**
1. **Sidebar — choose data source**: "Synthetic demo data" or "Import from Jira" (enter site URL/email/API token → list boards → pick sprint → import).
2. **View data**: Team table and Sprint Backlog table side by side.
3. **Run AI Sprint Planning** button → `POST /api/sprint/plan` → shows a **Recommended Assignments** table (sorted by fit score, with the `reason` string) and any unassigned tasks.
4. **Visualizations** (Plotly Express): a workload/capacity-utilization bar chart per employee, and a mean-fit-score-by-priority bar chart.
5. **Feedback loop**: an editable table pre-filled with predicted values lets the user record what *actually* happened (`actual_success`, `actual_days`) after the sprint — saved via `POST /api/sprint/outcomes`, feeding the data `train_model.py` can eventually retrain on.

For Jira-imported data, since Jira has no skill/experience/capacity fields, the dashboard shows an editable table (pre-filled from any previously saved `EmployeeProfile`) so the user fills those in once and it's remembered for next time.

---

## 5. Dependencies & Why Each Was Chosen

### Backend (`backend/requirements.txt`)
| Package | Why |
|---|---|
| `fastapi` | Modern async Python web framework with automatic OpenAPI docs (`/docs`) and native Pydantic integration |
| `uvicorn[standard]` | ASGI server to actually run FastAPI |
| `pydantic` | Request/response validation — catches malformed data at the API boundary instead of deep in the ML code |
| `scikit-learn` | RandomForest models, `OneHotEncoder`, `Pipeline`, `ColumnTransformer`, metrics — the whole training/inference stack |
| `scipy` | `linear_sum_assignment` — the Hungarian algorithm implementation for optimal assignment |
| `pandas` | Dataframe manipulation for CSV I/O, feature building, and the optimizer's tabular data |
| `numpy` | Random distributions and array ops in synthetic data generation and the optimizer's cost matrix |
| `joblib` | Serialize/deserialize trained sklearn pipelines to disk |
| `faker` | Generates realistic fake names/text for the synthetic dataset |
| `requests` | HTTP client for calling the Jira Cloud REST API |
| `sqlalchemy` | ORM for the employee-profile and outcome-logging tables, DB-agnostic (SQLite now, Postgres-ready) |
| `psycopg2-binary` | Postgres driver, only used if `DATABASE_URL` is pointed at Postgres |

### Frontend (`frontend/requirements.txt`)
| Package | Why |
|---|---|
| `streamlit` | Fast way to build a data-app UI in pure Python — no separate JS frontend needed |
| `requests` | Calls the FastAPI backend |
| `pandas` | Builds/merges the tables shown in the UI |
| `plotly` (via `plotly.express`) | The capacity-utilization and fit-by-priority bar charts |

---

## 6. Configuration

- **No `.env` file** — environment variables are set directly in the shell before launching.
- **`DATABASE_URL`** (optional env var, read in `backend/app/db/database.py`): overrides the default SQLite path. Example: `postgresql+psycopg2://user:password@localhost:5432/sprint_ai`.
- **No separate settings module** — paths and constants (`DATA_DIR`, `API_BASE`) are inlined at the top of the relevant files.
- **Jira credentials** are never stored anywhere server-side — supplied per-request from the dashboard sidebar and passed straight through to `JiraClient`.
- **`.claude/launch.json`** — a dev-only convenience config (added during setup here) that lets a launcher start `backend` (uvicorn on port 8000) and `frontend` (streamlit on port 8501) by name, each pre-configured with the correct working directory and interpreter.

---

## 7. How to Run the Project — Step by Step

### One-time setup (already done in this environment, but for reference)
```bash
cd "C:\New folder\sharoline_ml_project\backend"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m app.data.generate_synthetic_data   # creates data/employees.csv, tasks.csv, history.csv
python -m app.ml.train_model                 # trains and saves fit_model.joblib, days_model.joblib
```
```bash
cd "C:\New folder\sharoline_ml_project\frontend"
pip install -r requirements.txt              # can reuse the backend venv's interpreter instead
```

### Every time you want to run it — two terminals, both must stay open

**Terminal 1 — backend** (must `cd` into `backend` first — it's a relative-import Python package):
```bash
cd "C:\New folder\sharoline_ml_project\backend"
venv\Scripts\activate
uvicorn app.main:app --reload
```
Wait for `Application startup complete.` The API is now at **http://localhost:8000** (interactive docs at **http://localhost:8000/docs**).

**Terminal 2 — frontend**:
```bash
cd "C:\New folder\sharoline_ml_project\frontend"
"C:\New folder\sharoline_ml_project\backend\venv\Scripts\python.exe" -m streamlit run dashboard.py
```
Wait for `Local URL: http://localhost:8501`. Open that URL in your browser — that's the app.

**To stop:** `Ctrl+C` in each terminal.

### Common pitfalls
- Running `uvicorn app.main:app` from the project root instead of `backend/` → `ModuleNotFoundError: No module named 'app'`.
- Running `streamlit run dashboard.py` from anywhere other than `frontend/` → file not found.
- If PowerShell blocks `venv\Scripts\activate` with an execution-policy error, either run `powershell -ExecutionPolicy Bypass -File venv\Scripts\Activate.ps1`, or skip activation entirely and call `venv\Scripts\uvicorn.exe` / `venv\Scripts\python.exe` directly.
- If you regenerate data or want a fresh model: re-run `python -m app.data.generate_synthetic_data` then `python -m app.ml.train_model` from inside `backend` with the venv active.

---

## 8. Known Gaps / Next Steps (from the README, still accurate)
- Swap synthetic data for a real Jira export by default (Jira import already exists but requires manual credentials each session).
- Feedback loop exists (`assignment_outcomes` table + UI) but `train_model.py` still trains on synthetic `history.csv` by default — wiring it to retrain on real logged outcomes is the natural next step.
- No authentication; CORS is wide open (`allow_origins=["*"]`) — fine for localhost, must be locked down before any real deployment.
- SQLite is the default DB; `DATABASE_URL` already supports switching to Postgres for production.
