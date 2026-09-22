"""
Endpoints backing the dashboard's "Connect Jira" panel. Credentials are
accepted in the request body and used only to call the user's own Jira
site for that single request — nothing is persisted on the backend.
"""
import requests as requests_lib
from fastapi import APIRouter, HTTPException

from app.integrations.jira_client import JiraAuthError, JiraClient
from app.models.schema import (
    ImportedEmployee, ImportedTask, JiraBoardsRequest, JiraImportRequest,
    JiraImportResponse, JiraSprintsRequest,
)

router = APIRouter()


def _client(creds) -> JiraClient:
    return JiraClient(creds.site_url, creds.email, creds.api_token)


def _handle_jira_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except JiraAuthError as e:
        raise HTTPException(401, str(e))
    except requests_lib.exceptions.RequestException as e:
        raise HTTPException(502, f"Could not reach Jira: {e}")


@router.post("/boards")
def get_boards(payload: JiraBoardsRequest):
    client = _client(payload)
    return _handle_jira_call(client.list_boards)


@router.post("/sprints")
def get_sprints(payload: JiraSprintsRequest):
    client = _client(payload)
    return _handle_jira_call(client.list_sprints, payload.board_id)


@router.post("/import", response_model=JiraImportResponse)
def import_sprint(payload: JiraImportRequest):
    client = _client(payload)

    issues = _handle_jira_call(client.get_sprint_issues, payload.board_id, payload.sprint_id)
    tasks = [ImportedTask(
        task_id=i["task_id"],
        title=i["title"],
        description=i["description"] or i["title"],
        task_type=_normalize_task_type(i["task_type"]),
        story_points=i["story_points"],
        priority=i["priority"],
        assignee_account_id=i["assignee_account_id"],
        assignee_name=i["assignee_name"],
    ) for i in issues]

    project_key = _handle_jira_call(client.get_board_project_key, payload.board_id)
    employees = []
    if project_key:
        users = _handle_jira_call(client.get_assignable_users, project_key)
        employees = [ImportedEmployee(**u) for u in users]

    return JiraImportResponse(employees=employees, tasks=tasks)


def _normalize_task_type(jira_issue_type: str) -> str:
    mapping = {
        "story": "feature", "task": "feature", "bug": "bug",
        "sub-task": "feature", "epic": "feature",
    }
    return mapping.get(jira_issue_type.lower(), "feature")
