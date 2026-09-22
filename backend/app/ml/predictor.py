"""Loads the trained models and scores every (employee, task) pair."""
import os
from typing import List

import joblib
import pandas as pd

from app.data.generate_synthetic_data import skill_match
from app.ml.skill_extractor import extract_required_skills
from app.models.schema import Employee, Task

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

CATEGORICAL_FEATURES = ["task_type"]
NUMERIC_FEATURES = ["skill_match", "experience_years", "velocity_avg",
                     "workload_at_assignment", "story_points"]


class FitPredictor:
    def __init__(self):
        fit_path = os.path.join(DATA_DIR, "fit_model.joblib")
        days_path = os.path.join(DATA_DIR, "days_model.joblib")
        if not os.path.exists(fit_path) or not os.path.exists(days_path):
            raise FileNotFoundError(
                "Trained models not found. Run: python -m app.ml.train_model"
            )
        self.fit_model = joblib.load(fit_path)
        self.days_model = joblib.load(days_path)

    def score_pairs(self, employees: List[Employee], tasks: List[Task]) -> pd.DataFrame:
        """Returns a long dataframe with one row per (employee, task) pair,
        including predicted success probability (fit_score) and predicted
        completion days."""
        rows = []
        for task in tasks:
            required = task.required_skills or extract_required_skills(task.description)
            for emp in employees:
                match = skill_match(emp.skills, required)
                rows.append({
                    "employee_id": emp.employee_id,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "skill_match": match,
                    "experience_years": emp.experience_years,
                    "velocity_avg": emp.velocity_avg,
                    "workload_at_assignment": emp.current_load_points,
                    "story_points": task.story_points,
                })
        pairs = pd.DataFrame(rows)
        X = pairs[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

        pairs["fit_score"] = self.fit_model.predict_proba(X)[:, 1]
        pairs["predicted_days"] = self.days_model.predict(X)
        return pairs
