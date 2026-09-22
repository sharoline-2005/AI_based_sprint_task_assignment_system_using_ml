from typing import List

from sqlalchemy.orm import Session

from app.db.models import AssignmentOutcome, EmployeeProfile


def upsert_employee_profile(db: Session, profile: dict) -> EmployeeProfile:
    existing = db.get(EmployeeProfile, profile["employee_id"])
    if existing:
        for k, v in profile.items():
            setattr(existing, k, v)
        row = existing
    else:
        row = EmployeeProfile(**profile)
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_employee_profiles(db: Session, employee_ids: List[str] = None) -> List[EmployeeProfile]:
    q = db.query(EmployeeProfile)
    if employee_ids:
        q = q.filter(EmployeeProfile.employee_id.in_(employee_ids))
    return q.all()


def log_outcome(db: Session, outcome: dict) -> AssignmentOutcome:
    row = AssignmentOutcome(**outcome)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_outcomes(db: Session) -> List[AssignmentOutcome]:
    return db.query(AssignmentOutcome).all()
