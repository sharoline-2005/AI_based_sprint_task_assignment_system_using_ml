"""
Thin client for the Jira Cloud REST API. Credentials are supplied per-call
by the caller (never stored on the backend) and used only to reach the
user's own Jira site with Basic Auth (email + API token, Atlassian's
documented auth method for Cloud).
"""
from typing import Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

TIMEOUT = 15


class JiraAuthError(Exception):
    pass


class JiraClient:
    def __init__(self, site_url: str, email: str, api_token: str):
        self.base = site_url.rstrip("/")
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {"Accept": "application/json"}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = requests.get(
            f"{self.base}{path}", auth=self.auth, headers=self.headers,
            params=params, timeout=TIMEOUT,
        )
        if resp.status_code in (401, 403):
            raise JiraAuthError("Invalid Jira credentials or insufficient permissions.")
        resp.raise_for_status()
        return resp.json()

    def test_connection(self) -> dict:
        return self._get("/rest/api/3/myself")

    def list_boards(self) -> List[dict]:
        data = self._get("/rest/agile/1.0/board", params={"maxResults": 50})
        return [{"id": b["id"], "name": b["name"], "type": b.get("type")}
                for b in data.get("values", [])]

    def get_board_project_key(self, board_id: int) -> Optional[str]:
        data = self._get(f"/rest/agile/1.0/board/{board_id}/configuration")
        return (data.get("location") or {}).get("projectKey")

    def list_sprints(self, board_id: int) -> List[dict]:
        data = self._get(f"/rest/agile/1.0/board/{board_id}/sprint",
                          params={"state": "active,future", "maxResults": 50})
        return [{"id": s["id"], "name": s["name"], "state": s["state"]}
                for s in data.get("values", [])]

    def _story_points_field_id(self) -> Optional[str]:
        fields = self._get("/rest/api/3/field")
        for f in fields:
            if f.get("name", "").strip().lower() == "story points":
                return f["id"]
        return None

    def get_sprint_issues(self, board_id: int, sprint_id: int) -> List[dict]:
        sp_field = self._story_points_field_id()
        fields_param = "summary,description,issuetype,assignee,priority"
        if sp_field:
            fields_param += f",{sp_field}"

        issues, start_at = [], 0
        while True:
            data = self._get(
                f"/rest/agile/1.0/board/{board_id}/sprint/{sprint_id}/issue",
                params={"fields": fields_param, "startAt": start_at, "maxResults": 50},
            )
            issues.extend(data.get("issues", []))
            if start_at + 50 >= data.get("total", 0):
                break
            start_at += 50

        out = []
        for issue in issues:
            f = issue["fields"]
            assignee = f.get("assignee") or {}
            desc = f.get("description")
            desc_text = _extract_text(desc) if isinstance(desc, dict) else (desc or "")
            out.append({
                "task_id": issue["key"],
                "title": f.get("summary", ""),
                "description": desc_text,
                "task_type": (f.get("issuetype") or {}).get("name", "feature").lower(),
                "story_points": f.get(sp_field) if sp_field else None,
                "priority": _priority_rank((f.get("priority") or {}).get("name")),
                "assignee_account_id": assignee.get("accountId"),
                "assignee_name": assignee.get("displayName"),
            })
        return out

    def get_assignable_users(self, project_key: str) -> List[dict]:
        data = self._get(
            "/rest/api/3/user/assignable/search",
            params={"project": project_key, "maxResults": 50},
        )
        return [{
            "employee_id": u["accountId"],
            "name": u.get("displayName", "Unknown"),
        } for u in data]


def _priority_rank(name: Optional[str]) -> int:
    order = {"highest": 1, "high": 2, "medium": 3, "low": 4, "lowest": 5}
    return order.get((name or "").lower(), 3)


def _extract_text(adf_doc: Dict) -> str:
    """Flattens Jira's Atlassian Document Format description into plain text."""
    parts = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "text":
                parts.append(node.get("text", ""))
            for child in node.get("content", []) or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(adf_doc)
    return " ".join(parts)
