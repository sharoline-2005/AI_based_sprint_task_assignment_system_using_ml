import json
import os
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.crud import get_employee_profiles, get_outcomes, log_outcome, upsert_employee_profile
from app.db.database import get_db
from app.ml.optimizer import optimize_assignment
from app.ml.predictor import FitPredictor
from app.models.schema import (
    Employee, EmployeeProfileIn, OutcomeIn, Task, SprintPlanRequest, SprintPlanResponse,
)

router = APIRouter()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

_predictor: Optional[FitPredictor] = None


def get_predictor() -> FitPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FitPredictor()
    return _predictor


def _load_employees_from_csv() -> List[Employee]:
    df = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
    out = []
    for _, row in df.iterrows():
        out.append(Employee(
            employee_id=row["employee_id"],
            name=row["name"],
            role=row["role"],
            experience_years=row["experience_years"],
            skills=json.loads(row["skills"]),
            velocity_avg=row["velocity_avg"],
            capacity_points=row["capacity_points"],
            current_load_points=row["current_load_points"],
        ))
    return out


def _load_tasks_from_csv() -> List[Task]:
    df = pd.read_csv(os.path.join(DATA_DIR, "tasks.csv"))
    out = []
    for _, row in df.iterrows():
        out.append(Task(
            task_id=row["task_id"],
            sprint_id=row["sprint_id"],
            title=row["title"],
            description=row["description"],
            story_points=row["story_points"],
            task_type=row["task_type"],
            priority=int(row["priority"]),
            required_skills=json.loads(row["required_skills"]),
        ))
    return out


@router.get("/employees", response_model=List[Employee])
def list_employees():
    try:
        return _load_employees_from_csv()
    except FileNotFoundError:
        raise HTTPException(404, "No employee data. Run the data generator first.")


@router.get("/tasks", response_model=List[Task])
def list_tasks():
    try:
        return _load_tasks_from_csv()
    except FileNotFoundError:
        raise HTTPException(404, "No task data. Run the data generator first.")


@router.post("/sprint/plan", response_model=SprintPlanResponse)
def plan_sprint(payload: Optional[SprintPlanRequest] = None):
    """
    Runs the full pipeline: skill matching -> ML fit prediction ->
    Hungarian-algorithm optimization -> assignment plan.

    If no payload is given, falls back to the synthetic CSVs in data/ so the
    endpoint is demoable via GET-like calls from the dashboard/Swagger UI.
    """
    if payload is not None:
        employees, tasks = payload.employees, payload.tasks
    else:
        employees, tasks = _load_employees_from_csv(), _load_tasks_from_csv()

    if not employees or not tasks:
        raise HTTPException(400, "Need at least one employee and one task.")

    predictor = get_predictor()
    pairs = predictor.score_pairs(employees, tasks)
    assignments, unassigned = optimize_assignment(employees, tasks, pairs)

    return SprintPlanResponse(assignments=assignments, unassigned_task_ids=unassigned)


@router.post("/employees/profile")
def save_employee_profile(payload: EmployeeProfileIn, db: Session = Depends(get_db)):
    """Persists a skill/experience profile so it doesn't need re-entering on the next import."""
    row = upsert_employee_profile(db, payload.model_dump())
    return {"employee_id": row.employee_id, "saved": True}


@router.get("/employees/profiles")
def list_employee_profiles(db: Session = Depends(get_db)):
    """Returns all saved profiles, keyed by employee_id, for the dashboard to pre-fill with."""
    rows = get_employee_profiles(db)
    return {
        r.employee_id: {
            "role": r.role, "experience_years": r.experience_years,
            "velocity_avg": r.velocity_avg, "capacity_points": r.capacity_points,
            "skills": r.skills,
        } for r in rows
    }


@router.post("/sprint/outcomes")
def log_sprint_outcome(payload: OutcomeIn, db: Session = Depends(get_db)):
    """
    Feedback-loop endpoint: call this once a sprint is actually run, with
    what really happened. app/ml/train_model.py can later be pointed at
    this table instead of the synthetic history.csv to retrain on real data.
    """
    row = log_outcome(db, payload.model_dump())
    return {"id": row.id, "logged": True}


@router.get("/sprint/outcomes")
def list_sprint_outcomes(db: Session = Depends(get_db)):
    rows = get_outcomes(db)
    return [{
        "sprint_id": r.sprint_id, "task_id": r.task_id, "employee_id": r.employee_id,
        "task_type": r.task_type, "story_points": r.story_points,
        "predicted_fit_score": r.predicted_fit_score, "predicted_days": r.predicted_days,
        "actual_success": bool(r.actual_success), "actual_days": r.actual_days,
        "logged_at": r.logged_at.isoformat(),
    } for r in rows]
