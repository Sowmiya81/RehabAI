from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AnalysisRecord(BaseModel):
    session_id: str
    timestamp: datetime
    video_filename: str
    issue_count: int
    coaching_plan_preview: str
    evidence_sources: int


class BiomechanicsIssue(BaseModel):
    joint: str
    description: str
    severity: str  # "low", "medium", "high"
    angle: Optional[float] = None
    recommendation: Optional[str] = None

