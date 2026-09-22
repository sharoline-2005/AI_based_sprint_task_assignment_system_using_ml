"""
Optimization layer: turns per-pair fit scores into a team-wide assignment
plan using the Hungarian Algorithm (scipy.optimize.linear_sum_assignment),
then enforces exact story-point capacity so no one is over-allocated.

Stage 1 (global optimum, approximate capacity): each employee is expanded
into N "capacity slots" (N = how many average-sized backlog tasks fit in
their remaining capacity). The Hungarian algorithm solves the resulting
slots x tasks assignment problem, maximizing total predicted fit.

Stage 2 (exact capacity enforcement): slot counts are only an average-based
estimate, so a slot-eligible employee can still end up with more real story
points than they have capacity for (e.g. several large tasks landing on the
same person). This stage walks the Stage-1 result best-fit-first, keeps an
assignment only if it fits the employee's *actual* remaining story-point
capacity, and re-routes anything that doesn't fit to the next best employee
who still has room.
"""
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from app.models.schema import Assignment, Employee, Task


def _capacity_slots(employee: Employee, avg_story_points: float) -> int:
    remaining = max(employee.capacity_points - employee.current_load_points, 0)
    avg_story_points = max(avg_story_points, 1)
    return max(1, math.floor(remaining / avg_story_points))


def _stage1_candidates(
    employees: List[Employee], tasks: List[Task], pairs: pd.DataFrame
) -> List[str]:
    """Returns, per task, the Stage-1 (capacity-ignorant) preferred employee_id."""
    avg_points = sum(t.story_points for t in tasks) / len(tasks)

    slot_employee_ids: List[str] = []
    for emp in employees:
        slots = _capacity_slots(emp, avg_points)
        slot_employee_ids.extend([emp.employee_id] * slots)

    task_ids = [t.task_id for t in tasks]
    lookup = pairs.set_index(["employee_id", "task_id"])
    cost = np.ones((len(slot_employee_ids), len(task_ids)))
    for r, emp_id in enumerate(slot_employee_ids):
        for c, task_id in enumerate(task_ids):
            cost[r, c] = 1.0 - float(lookup.loc[(emp_id, task_id), "fit_score"])

    row_idx, col_idx = linear_sum_assignment(cost)
    preferred = {}
    for r, c in zip(row_idx, col_idx):
        preferred[task_ids[c]] = slot_employee_ids[r]
    return [preferred.get(tid) for tid in task_ids]


def optimize_assignment(
    employees: List[Employee], tasks: List[Task], pairs: pd.DataFrame
) -> Tuple[List[Assignment], List[str]]:
    if not employees or not tasks:
        return [], [t.task_id for t in tasks]

    preferred_by_task = dict(zip([t.task_id for t in tasks], _stage1_candidates(employees, tasks, pairs)))

    remaining_capacity: Dict[str, float] = {
        emp.employee_id: max(emp.capacity_points - emp.current_load_points, 0)
        for emp in employees
    }
    story_points = {t.task_id: t.story_points for t in tasks}
    fit_lookup = pairs.set_index(["employee_id", "task_id"])["fit_score"].to_dict()
    days_lookup = pairs.set_index(["employee_id", "task_id"])["predicted_days"].to_dict()
    features_lookup = pairs.set_index(["employee_id", "task_id"])[
        ["skill_match", "experience_years", "velocity_avg", "workload_at_assignment"]
    ].to_dict("index")

    # process tasks best-fit-first so the strongest matches get first claim on capacity
    ordered_task_ids = sorted(
        [t.task_id for t in tasks],
        key=lambda tid: fit_lookup.get((preferred_by_task.get(tid), tid), 0.0),
        reverse=True,
    )

    assignments: List[Assignment] = []
    unassigned: List[str] = []

    for task_id in ordered_task_ids:
        points = story_points[task_id]
        preferred_emp = preferred_by_task.get(task_id)

        chosen_emp = None
        if preferred_emp and remaining_capacity.get(preferred_emp, 0) >= points:
            chosen_emp = preferred_emp
        else:
            # re-route to the best-fit employee who actually has room
            candidates = [
                (emp_id, fit_lookup.get((emp_id, task_id), 0.0))
                for emp_id, cap in remaining_capacity.items()
                if cap >= points
            ]
            if candidates:
                chosen_emp = max(candidates, key=lambda x: x[1])[0]

        if chosen_emp is None:
            unassigned.append(task_id)
            continue

        fit = float(fit_lookup.get((chosen_emp, task_id), 0.0))
        days = float(days_lookup.get((chosen_emp, task_id), 0.0))
        features = features_lookup.get((chosen_emp, task_id), {})
        remaining_capacity[chosen_emp] -= points

        assignments.append(Assignment(
            task_id=task_id,
            employee_id=chosen_emp,
            fit_score=round(fit, 3),
            predicted_days=round(days, 1),
            reason=(
                f"Predicted {fit*100:.0f}% success probability, "
                f"~{days:.1f} days to complete"
            ),
            skill_match=float(features.get("skill_match", 0.0)),
            experience_years=float(features.get("experience_years", 0.0)),
            velocity_avg=float(features.get("velocity_avg", 0.0)),
            workload_at_assignment=float(features.get("workload_at_assignment", 0.0)),
        ))

    return assignments, unassigned
