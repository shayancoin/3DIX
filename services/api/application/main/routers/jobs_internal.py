"""
Internal API routes for job processing.
These endpoints are used by the job worker and internal services.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from application.main.services.job_worker import JobWorker
from application.main.services.layout_service import LayoutService

router = APIRouter()
layout_service = LayoutService()

# Global worker instance (in production, use a proper task queue)
_worker: Optional[JobWorker] = None


def get_worker() -> JobWorker:
    """Get or create worker instance."""
    global _worker
    if _worker is None:
        _worker = JobWorker()
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
async def get_queued_jobs():
    """
    Get queued jobs for processing.
    Note: In a real implementation, this would query the database.
    """
    # TODO: Implement database query
    return []
