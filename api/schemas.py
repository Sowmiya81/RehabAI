# api/schemas.py
from pydantic import BaseModel
from typing import Any


class AnalyzeResponse(BaseModel):
    issues: list[dict[str, Any]]
    metrics: dict[str, Any]
    coaching_plan: str
    evidence: list[dict[str, Any]]
    issue_count: int
    evidence_count: int


class ErrorResponse(BaseModel):
    detail: str
    status_code: int
