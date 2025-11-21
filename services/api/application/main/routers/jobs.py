"""
API routes for job management.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from application.main.infrastructure.database.postgresql.operations import Postgresql

router = APIRouter()


def get_db() -> Postgresql:
    """Dependency to get PostgreSQL database instance."""
    return Postgresql()


class JobResponse(BaseModel):
    id: int
    room_id: int
    status: str
    progress: Optional[int]
    progress_message: Optional[str]
    request_data: Optional[dict]
    response_data: Optional[dict]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.get("/queued", response_model=List[JobResponse])
async def get_queued_jobs(db: Postgresql = Depends(get_db)):
    """
    Get all queued jobs.
    """
    try:
        jobs = await db.get_queued_jobs()
        return [JobResponse(**job) for job in jobs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching queued jobs: {str(e)}")


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, db: Postgresql = Depends(get_db)):
    """
    Get a specific job by ID.
    """
    try:
        job = await db.get_layout_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(**job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job: {str(e)}")


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(job_id: int, updates: dict, db: Postgresql = Depends(get_db)):
    """
    Update a job.
    """
    try:
        updated_job = await db.update_layout_job(job_id, updates)
        if not updated_job:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(**updated_job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating job: {str(e)}")
