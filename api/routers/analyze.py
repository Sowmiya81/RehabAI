from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Any
from pathlib import Path
import tempfile, shutil, os, sys

from api.schemas import AnalyzeResponse, ErrorResponse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import RehabCoachAgent

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Populated at startup via api/main.py lifespan
_agent: RehabCoachAgent | None = None


def init_agent(max_steps: int = 3) -> None:
    """Called once at app startup — not per request."""
    global _agent
    _agent = RehabCoachAgent(max_steps=max_steps)


def get_agent() -> RehabCoachAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")
    return _agent


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyze exercise video for biomechanical issues",
)
async def analyze_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    agent = get_agent()
    tmp_dir = tempfile.mkdtemp(prefix="rehabai_api_")
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        result: dict[str, Any] = agent.run(tmp_path)

        biomech = result.get("biomechanics") or {}
        issues = biomech.get("issues") or []
        metrics = {k: v for k, v in biomech.items() if k != "issues"}
        evidence = result.get("evidence") or []
        coaching = result.get("coaching_plan") or ""

        return AnalyzeResponse(
            issues=issues,
            metrics=metrics,
            coaching_plan=coaching,
            evidence=evidence,
            issue_count=len(issues),
            evidence_count=len(evidence),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "RehabAI API", "agent_ready": _agent is not None}
