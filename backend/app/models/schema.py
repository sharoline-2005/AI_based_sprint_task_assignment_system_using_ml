"""Shared skill taxonomy and Pydantic schemas used across the pipeline."""
from typing import Dict, List, Optional
from pydantic import BaseModel

# Fixed skill vocabulary. Every employee and task is scored against this list.
SKILLS: List[str] = [
    "python", "java", "javascript", "react", "node", "sql", "aws",
    "docker", "kubernetes", "testing", "ui_ux", "devops",
    "machine_learning", "android", "ios", "spring_boot", "django",
]

TASK_TYPES: List[str] = ["feature", "bug", "testing", "devops", "research"]


class Employee(BaseModel):
    employee_id: str
    name: str
    role: str
    experience_years: float
    skills: Dict[str, float]          # skill -> proficiency 0-1
    velocity_avg: float               # avg story points completed per sprint
    capacity_points: float            # story points available this sprint
    current_load_points: float = 0.0  # points already assigned this sprint


class Task(BaseModel):
    task_id: str
    sprint_id: str
    title: str
    description: str
    story_points: float
    task_type: str
    priority: int = 3                 # 1 (highest) - 5 (lowest)
    required_skills: Optional[Dict[str, float]] = None  # filled by extractor if omitted


class Assignment(BaseModel):
    task_id: str
    employee_id: str
    fit_score: float                  # predicted success probability (0-1)
    predicted_days: float
    reason: str
    # features used for this pair's prediction, carried through so the
    # feedback-loop outcome log can be trained on later without recomputing them
    skill_match: float
    experience_years: float
    velocity_avg: float
    workload_at_assignment: float


class SprintPlanRequest(BaseModel):
    employees: List[Employee]
    tasks: List[Task]


class SprintPlanResponse(BaseModel):
    assignments: List[Assignment]
    unassigned_task_ids: List[str]


class JiraCredentials(BaseModel):
    site_url: str            # e.g. https://yourcompany.atlassian.net
    email: str
    api_token: str


class JiraBoardsRequest(JiraCredentials):
    pass


class JiraSprintsRequest(JiraCredentials):
    board_id: int


class JiraImportRequest(JiraCredentials):
    board_id: int
    sprint_id: int


class ImportedEmployee(BaseModel):
    employee_id: str
    name: str


class ImportedTask(BaseModel):
    task_id: str
    title: str
    description: str
    task_type: str
    story_points: Optional[float] = None
    priority: int
    assignee_account_id: Optional[str] = None
    assignee_name: Optional[str] = None


class JiraImportResponse(BaseModel):
    employees: List[ImportedEmployee]
    tasks: List[ImportedTask]


class EmployeeProfileIn(BaseModel):
    employee_id: str
    name: str
    role: str = "Team Member"
    experience_years: float = 2.0
    velocity_avg: float = 8.0
    capacity_points: float = 10.0
    skills: Dict[str, float] = {}


class OutcomeIn(BaseModel):
    sprint_id: str
    task_id: str
    employee_id: str
    task_type: str
    story_points: float
    predicted_fit_score: float
    predicted_days: float
    actual_success: bool
    actual_days: float
    # same features the original prediction was made from -- required so this
    # outcome can be blended back into training data (see app/ml/train_model.py)
    skill_match: float
    experience_years: float
    velocity_avg: float
    workload_at_assignment: float
