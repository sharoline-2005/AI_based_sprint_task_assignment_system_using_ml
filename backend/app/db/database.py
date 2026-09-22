"""
Persistence layer. Defaults to a local SQLite file so the project runs with
zero extra infra, but the exact same SQLAlchemy models/code work unchanged
against a real Postgres server -- just set DATABASE_URL, e.g.:

    postgresql+psycopg2://user:password@localhost:5432/sprint_ai

before starting the API (a .env file or `set`/`export` both work).
"""
import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DEFAULT_SQLITE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'app.db')}"

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.db import models  # noqa: F401 (registers tables with Base)
    Base.metadata.create_all(bind=engine)
    _migrate_add_missing_columns()


def _migrate_add_missing_columns():
    """
    Lightweight migration for columns added after a table already existed
    (e.g. an existing SQLite/Postgres DB from before the retrain-on-real-
    outcomes feature). create_all() only creates missing *tables*, not
    missing *columns* on existing tables, so add them here if absent.
    """
    from app.db.models import AssignmentOutcome

    inspector = inspect(engine)
    if AssignmentOutcome.__tablename__ not in inspector.get_table_names():
        return
    existing_cols = {c["name"] for c in inspector.get_columns(AssignmentOutcome.__tablename__)}
    for column in AssignmentOutcome.__table__.columns:
        if column.name not in existing_cols:
            col_type = column.type.compile(engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(
                    f'ALTER TABLE {AssignmentOutcome.__tablename__} '
                    f'ADD COLUMN {column.name} {col_type}'
                ))
