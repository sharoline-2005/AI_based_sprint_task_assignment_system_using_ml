from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.jira_routes import router as jira_router
from app.db.database import init_db

app = FastAPI(
    title="Sprint Assignment AI",
    description="Predicts employee-task fit and optimizes sprint task assignment.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(jira_router, prefix="/api/jira", tags=["jira"])


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
