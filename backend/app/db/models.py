import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class EmployeeProfile(Base):
    """
    Persisted skill/experience/capacity profile for a person, keyed by their
    Jira account ID (or synthetic employee_id). Saved once, reused for every
    future sprint import instead of re-entering skills each time.
    """
    __tablename__ = "employee_profiles"

    employee_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String, default="Team Member")
    experience_years: Mapped[float] = mapped_column(Float, default=2.0)
    velocity_avg: Mapped[float] = mapped_column(Float, default=8.0)
    capacity_points: Mapped[float] = mapped_column(Float, default=10.0)
    skills: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


class AssignmentOutcome(Base):
    """
    Logged result of a sprint assignment the AI recommended, once the sprint
    is actually run. This is the feedback-loop table: app/ml/train_model.py
    can be pointed at this table (instead of the synthetic history.csv) to
    retrain the fit-prediction model on real outcomes over time.
    """
    __tablename__ = "assignment_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sprint_id: Mapped[str] = mapped_column(String)
    task_id: Mapped[str] = mapped_column(String)
    employee_id: Mapped[str] = mapped_column(String)
    task_type: Mapped[str] = mapped_column(String)
    story_points: Mapped[float] = mapped_column(Float)
    predicted_fit_score: Mapped[float] = mapped_column(Float)
    predicted_days: Mapped[float] = mapped_column(Float)
    actual_success: Mapped[bool] = mapped_column(Integer)   # 0/1
    actual_days: Mapped[float] = mapped_column(Float)
    # same features the original prediction used, so this row can be blended
    # back into training data (see app/ml/train_model.py). Nullable because
    # outcomes logged before this column existed won't have them.
    skill_match: Mapped[float] = mapped_column(Float, nullable=True)
    experience_years: Mapped[float] = mapped_column(Float, nullable=True)
    velocity_avg: Mapped[float] = mapped_column(Float, nullable=True)
    workload_at_assignment: Mapped[float] = mapped_column(Float, nullable=True)
    logged_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
