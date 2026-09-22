"""
Trains two models on data/history.csv:
  1. RandomForestClassifier -> P(success) given (skill_match, experience,
     velocity, workload, story_points, task_type)
  2. RandomForestRegressor  -> expected completion_days

Both are saved to data/fit_model.joblib / data/days_model.joblib for the
API to load at request time.

Run:  python -m app.ml.train_model
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.models.schema import TASK_TYPES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

NUMERIC_FEATURES = ["skill_match", "experience_years", "velocity_avg",
                     "workload_at_assignment", "story_points"]
CATEGORICAL_FEATURES = ["task_type"]


def build_pipeline(estimator):
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[TASK_TYPES], handle_unknown="ignore"),
         CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    return Pipeline([("prep", preprocess), ("model", estimator)])


def main():
    history_path = os.path.join(DATA_DIR, "history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            "data/history.csv not found. Run: python -m app.data.generate_synthetic_data"
        )

    df = pd.read_csv(history_path)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_success = df["success"]
    y_days = df["completion_days"]

    X_train, X_test, ys_train, ys_test, yd_train, yd_test = train_test_split(
        X, y_success, y_days, test_size=0.2, random_state=42
    )

    fit_model = build_pipeline(
        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    )
    fit_model.fit(X_train, ys_train)
    auc = roc_auc_score(ys_test, fit_model.predict_proba(X_test)[:, 1])

    days_model = build_pipeline(
        RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    )
    days_model.fit(X_train, yd_train)
    mae = mean_absolute_error(yd_test, days_model.predict(X_test))

    joblib.dump(fit_model, os.path.join(DATA_DIR, "fit_model.joblib"))
    joblib.dump(days_model, os.path.join(DATA_DIR, "days_model.joblib"))

    print(f"Success-prediction model AUC: {auc:.3f}")
    print(f"Completion-days model MAE:   {mae:.2f} days")
    print("Saved -> data/fit_model.joblib, data/days_model.joblib")


if __name__ == "__main__":
    main()
