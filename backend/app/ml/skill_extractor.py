"""
Extracts required skills from a free-text task description when a task
doesn't already carry a `required_skills` dict (e.g. tasks pulled live from
Jira). Simple keyword/synonym matching against the fixed SKILLS vocabulary —
enough signal for the fit model without needing a trained NLP model.
"""
import re
from typing import Dict

from app.models.schema import SKILLS

SYNONYMS = {
    "js": "javascript",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node",
    "node.js": "node",
    "k8s": "kubernetes",
    "ml": "machine_learning",
    "ai": "machine_learning",
    "postgres": "sql",
    "mysql": "sql",
    "qa": "testing",
    "ux": "ui_ux",
    "ui": "ui_ux",
    "ci/cd": "devops",
}


def extract_required_skills(description: str) -> Dict[str, float]:
    text = description.lower()
    for syn, canonical in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(syn)}\b", canonical, text)

    found = {}
    for skill in SKILLS:
        keyword = skill.replace("_", " ")
        if re.search(rf"\b{re.escape(skill)}\b", text) or re.search(rf"\b{re.escape(keyword)}\b", text):
            found[skill] = 1.0

    if not found:
        # no explicit mention -> neutral requirement, don't penalize any employee
        return {}
    return found
