"""
Generates a synthetic-but-realistic dataset for the sprint-assignment project:
  - employees.csv   : team skill/experience profiles
  - tasks.csv       : an open sprint backlog (unassigned, to be planned)
  - history.csv     : past (employee, task) assignments + outcomes, used to
                       train the fit-prediction model

Run:  python -m app.data.generate_synthetic_data
"""
import json
import os
import random

import numpy as np
import pandas as pd
from faker import Faker

from app.models.schema import SKILLS, TASK_TYPES

fake = Faker()
random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_EMPLOYEES = 25
N_HISTORICAL_TASKS = 3000
N_BACKLOG_TASKS = 20

ROLE_SKILL_BIAS = {
    "Backend Engineer": ["python", "java", "sql", "django", "spring_boot", "aws"],
    "Frontend Engineer": ["javascript", "react", "ui_ux"],
    "Full Stack Engineer": ["javascript", "react", "node", "python", "sql"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "devops"],
    "QA Engineer": ["testing", "python", "javascript"],
    "ML Engineer": ["python", "machine_learning", "sql"],
    "Mobile Engineer": ["android", "ios", "java"],
}


def random_skill_vector(primary_skills):
    """Employee is strong in their role's skills, weak/absent elsewhere."""
    vec = {}
    for skill in SKILLS:
        if skill in primary_skills:
            vec[skill] = round(random.uniform(0.6, 1.0), 2)
        elif random.random() < 0.35:
            vec[skill] = round(random.uniform(0.1, 0.5), 2)
    return vec


def skill_match(employee_skills: dict, required_skills: dict) -> float:
    """Weighted overlap between what a task needs and what the employee has."""
    if not required_skills:
        return 0.5
    num, den = 0.0, 0.0
    for skill, weight in required_skills.items():
        den += weight
        num += weight * employee_skills.get(skill, 0.0)
    return round(num / den, 3) if den > 0 else 0.5


def random_required_skills(primary_skills, k=None):
    k = k or random.randint(1, 3)
    chosen = random.sample(primary_skills, min(k, len(primary_skills)))
    return {s: round(random.uniform(0.5, 1.0), 2) for s in chosen}


def gen_employees():
    rows = []
    for i in range(N_EMPLOYEES):
        role = random.choice(list(ROLE_SKILL_BIAS.keys()))
        primary = ROLE_SKILL_BIAS[role]
        experience = round(np.clip(np.random.exponential(3.5), 0.5, 15), 1)
        velocity = round(np.clip(np.random.normal(8 + experience * 0.5, 2), 2, 25), 1)
        capacity = round(np.clip(np.random.normal(velocity, 1.5), 4, 30), 1)
        rows.append({
            "employee_id": f"E{i+1:03d}",
            "name": fake.name(),
            "role": role,
            "experience_years": experience,
            "skills": json.dumps(random_skill_vector(primary)),
            "velocity_avg": velocity,
            "capacity_points": capacity,
            "current_load_points": 0.0,
        })
    return pd.DataFrame(rows)


def gen_backlog_tasks():
    rows = []
    all_primary = [s for lst in ROLE_SKILL_BIAS.values() for s in lst]
    for i in range(N_BACKLOG_TASKS):
        task_type = random.choice(TASK_TYPES)
        role_ref = random.choice(list(ROLE_SKILL_BIAS.keys()))
        req = random_required_skills(ROLE_SKILL_BIAS[role_ref])
        rows.append({
            "task_id": f"T{i+1:03d}",
            "sprint_id": "SPRINT-CURRENT",
            "title": f"{task_type.title()}: {fake.catch_phrase()}",
            "description": f"Requires skills: {', '.join(req.keys())}. {fake.sentence()}",
            "story_points": random.choice([1, 2, 3, 5, 8, 13]),
            "task_type": task_type,
            "priority": random.randint(1, 5),
            "required_skills": json.dumps(req),
        })
    return pd.DataFrame(rows)


def gen_history(employees_df):
    """
    Simulates past sprints: for each historical task, pick a random employee
    who *could* have been assigned, and derive a realistic outcome from
    skill-match + experience + a bit of noise. This gives the model a real
    signal to learn (higher skill-match & experience -> higher success,
    lower completion time).
    """
    rows = []
    employees = employees_df.to_dict("records")

    for i in range(N_HISTORICAL_TASKS):
        task_type = random.choice(TASK_TYPES)
        role_ref = random.choice(list(ROLE_SKILL_BIAS.keys()))
        required = random_required_skills(ROLE_SKILL_BIAS[role_ref])
        story_points = random.choice([1, 2, 3, 5, 8, 13])

        emp = random.choice(employees)
        emp_skills = json.loads(emp["skills"])
        match = skill_match(emp_skills, required)
        experience = emp["experience_years"]
        velocity = emp["velocity_avg"]

        # success probability rises with skill match & experience, falls with task size
        logit = (
            -1.5
            + 3.2 * match
            + 0.12 * min(experience, 10)
            - 0.05 * story_points
            + np.random.normal(0, 0.25)
        )
        prob_success = 1 / (1 + np.exp(-logit))
        success = int(np.random.random() < prob_success)

        base_days = story_points * (1.8 - 0.9 * match) * (10 / max(velocity, 3))
        completion_days = round(max(0.5, base_days + np.random.normal(0, 0.6)), 1)

        rows.append({
            "employee_id": emp["employee_id"],
            "task_type": task_type,
            "story_points": story_points,
            "skill_match": match,
            "experience_years": experience,
            "velocity_avg": velocity,
            "workload_at_assignment": round(random.uniform(0, emp["capacity_points"]), 1),
            "success": success,
            "completion_days": completion_days,
        })

    return pd.DataFrame(rows)


def main():
    employees_df = gen_employees()
    tasks_df = gen_backlog_tasks()
    history_df = gen_history(employees_df)

    employees_df.to_csv(os.path.join(DATA_DIR, "employees.csv"), index=False)
    tasks_df.to_csv(os.path.join(DATA_DIR, "tasks.csv"), index=False)
    history_df.to_csv(os.path.join(DATA_DIR, "history.csv"), index=False)

    print(f"Generated {len(employees_df)} employees -> data/employees.csv")
    print(f"Generated {len(tasks_df)} backlog tasks -> data/tasks.csv")
    print(f"Generated {len(history_df)} historical assignments -> data/history.csv")


if __name__ == "__main__":
    main()
