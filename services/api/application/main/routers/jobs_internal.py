"""
Internal API routes for job processing.
These endpoints are used by the job worker and internal services.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from application.main.services.stub_worker import StubWorker
from application.main.infrastructure.database.postgresql.operations import Postgresql

router = APIRouter()

# Global worker instance (in production, use a proper task queue)
_worker: Optional[StubWorker] = None


def get_db() -> Postgresql:
    """Dependency to get PostgreSQL database instance."""
    return Postgresql()


def get_worker() -> StubWorker:
    """Get or create worker instance."""
    global _worker
    if _worker is None:
        _worker = StubWorker()
    return _worker


@router.post("/process/{job_id}")
async def process_job(job_id: int, request_data: dict):
    """
    Process a job (called by worker or background task).
    """
    worker = get_worker()
    try:
        result = await worker.process_job(job_id, request_data)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queued")
async def get_queued_jobs(db: Postgresql = Depends(get_db)):
    """
    Get queued jobs for processing.
    """
    try:
        jobs = await db.get_queued_jobs()
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching queued jobs: {str(e)}")
