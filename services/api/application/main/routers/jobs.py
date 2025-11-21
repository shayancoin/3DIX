"""
API routes for job management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from application.main.services.job_worker import JobWorker
from application.main.services.layout_service import LayoutService
from application.main.infrastructure.database.postgresql.operations import PostgreSQL

router = APIRouter()
layout_service = LayoutService()


async def get_db() -> PostgreSQL:
    """Dependency to get PostgreSQL database instance."""
    db = PostgreSQL()
    await db.initialize()
    return db


class JobCreate(BaseModel):
    room_id: int
    request_data: dict


class JobUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    progress_message: Optional[str] = None
    response_data: Optional[dict] = None
    error_message: Optional[str] = None
    error_details: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None


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


@router.post("/", response_model=JobResponse, status_code=201)
async def create_job(job: JobCreate, background_tasks: BackgroundTasks, db: PostgreSQL = Depends(get_db)):
    """
    Create a new layout generation job.
    """
    try:
        # Verify room exists
        # Note: We'd need to get project_id and team_id from room, but for now we'll skip that check
        # In production, add proper validation
        
        created = await db.create_layout_job(
            room_id=job.room_id,
            request_data=job.request_data,
            status="queued"
        )
        
        # Start background worker to process the job
        background_tasks.add_task(process_job_background, created["id"], job.request_data)
        
        return JobResponse(**created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


async def process_job_background(job_id: int, request_data: dict):
    """Background task to process a job."""
    worker = JobWorker()
    try:
        await worker.process_job(job_id, request_data)
    except Exception as e:
        # Error is already logged in worker
        pass


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Get a specific job by ID.
    """
    try:
        job = await db.get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(**job)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(job_id: int, updates: JobUpdate, db: PostgreSQL = Depends(get_db)):
    """
    Update a job.
    """
    try:
        updated = await db.update_job(
            job_id=job_id,
            status=updates.status,
            progress=updates.progress,
            progress_message=updates.progress_message,
            response_data=updates.response_data,
            error_message=updates.error_message,
            error_details=updates.error_details,
            started_at=updates.started_at,
            completed_at=updates.completed_at,
            worker_id=updates.worker_id
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(**updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
