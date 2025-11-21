"""
API routes for job management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from application.main.services.job_worker import JobWorker
from application.main.services.layout_service import LayoutService

router = APIRouter()
layout_service = LayoutService()


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
async def get_queued_jobs():
    """
    Get all queued jobs.
    Note: In a real implementation, this would query the database.
    """
    # TODO: Implement database query
    return []


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int):
    """
    Get a specific job by ID.
    """
    # TODO: Implement database query
    raise HTTPException(status_code=404, detail="Job not found")


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(job_id: int, updates: dict):
    """
    Update a job.
    """
    # TODO: Implement database update
    raise HTTPException(status_code=404, detail="Job not found")
