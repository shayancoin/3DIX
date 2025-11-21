"""
Worker for Step 4 that calls the ML microservice for layout generation.
This replaces the stub worker from Step 3.
"""

import asyncio
import time
import uuid
from typing import Optional
from datetime import datetime
import os
import httpx
from application.main.infrastructure.database.postgresql.operations import Postgresql


class StubWorker:
    """Worker that calls ML microservice for layout generation."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        db: Optional[Postgresql] = None,
        ml_service_url: Optional[str] = None
    ):
        self.db = db or Postgresql()
        self.ml_service_url = ml_service_url or os.getenv(
            "ML_SERVICE_URL",
            "http://localhost:8001"
        )
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.running = False
        self.current_job_id: Optional[int] = None

    async def call_ml_service(self, request_data: dict) -> dict:
        """
        Call the ML microservice to generate a layout.
        Falls back to stub generation if ML service is unavailable.
        """
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.ml_service_url}/generate",
                    json=request_data,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            print(f"ML service unavailable ({e}), falling back to stub generation")
            # Fallback to stub generation
            return self.generate_dummy_layout(request_data)
    
    def generate_dummy_layout(self, request_data: dict) -> dict:
        """
        Generate a dummy layout response as fallback.
        This is used when ML service is unavailable.
        """
        # Extract room type from request data
        vibe_spec = request_data.get('vibeSpec', {}) or request_data.get('vibe_spec', {})
        room_type = (
            request_data.get('roomType') or 
            request_data.get('room_type') or
            vibe_spec.get('prompt', {}).get('roomType') or
            'kitchen'
        )
        
        # Generate dummy objects based on room type
        objects = []
        if room_type == 'kitchen':
            objects = [
                {
                    "id": "refrigerator-1",
                    "category": "refrigerator",
                    "position": [0.5, 0, 0.5],
                    "size": [0.7, 1.8, 0.7],
                    "orientation": 0
                },
                {
                    "id": "sink-1",
                    "category": "sink",
                    "position": [2.5, 0, 1.0],
                    "size": [0.6, 0.8, 0.5],
                    "orientation": 0
                },
                {
                    "id": "stove-1",
                    "category": "stove",
                    "position": [4.0, 0, 1.0],
                    "size": [0.6, 0.9, 0.6],
                    "orientation": 0
                },
                {
                    "id": "cabinet-1",
                    "category": "cabinet",
                    "position": [1.0, 0, 0.2],
                    "size": [1.5, 0.9, 0.6],
                    "orientation": 0
                },
            ]
        elif room_type == 'bedroom':
            objects = [
                {
                    "id": "bed-1",
                    "category": "bed",
                    "position": [2.0, 0, 2.0],
                    "size": [2.0, 0.5, 1.8],
                    "orientation": 0
                },
                {
                    "id": "dresser-1",
                    "category": "dresser",
                    "position": [0.5, 0, 0.5],
                    "size": [1.2, 1.0, 0.5],
                    "orientation": 0
                },
                {
                    "id": "nightstand-1",
                    "category": "nightstand",
                    "position": [0.5, 0, 2.0],
                    "size": [0.5, 0.6, 0.5],
                    "orientation": 0
                },
            ]
        else:
            # Generic room
            objects = [
                {
                    "id": "table-1",
                    "category": "table",
                    "position": [2.5, 0, 2.0],
                    "size": [1.2, 0.75, 0.8],
                    "orientation": 0
                },
                {
                    "id": "chair-1",
                    "category": "chair",
                    "position": [2.0, 0, 1.5],
                    "size": [0.5, 0.9, 0.5],
                    "orientation": 0
                },
            ]

        return {
            "jobId": request_data.get('roomId', 'unknown'),
            "status": "completed",
            "objects": objects,
            "semanticMap": None,
            "mask": None,
            "metadata": {
                "stub": True,
                "fallback": True,
            }
        }

    async def process_job(self, job_id: int, request_data: dict) -> dict:
        """
        Process a layout generation job with stub logic.
        """
        self.current_job_id = job_id
        start_time = time.time()

        try:
            # Update job status to running
            await self.update_job_status(
                job_id,
                "running",
                progress=0,
                progress_message="Starting layout generation...",
                started_at=datetime.utcnow()
            )

            # Format request for ML service
            # The request_data from frontend should already be in the correct format
            # but we ensure it matches LayoutRequest schema
            ml_request = {
                "roomId": request_data.get("roomId", str(job_id)),
                "vibeSpec": request_data.get("vibeSpec", {}),
                "constraints": request_data.get("constraints", {}),
            }

            await self.update_job_status(
                job_id,
                "running",
                progress=20,
                progress_message="Calling ML service..."
            )

            # Call ML service to generate layout
            ml_response = await self.call_ml_service(ml_request)
            
            await self.update_job_status(
                job_id,
                "running",
                progress=90,
                progress_message="Processing ML response..."
            )

            # Format response data to match expected structure
            response_data = {
                "jobId": str(job_id),
                "status": ml_response.get("status", "completed"),
                "objects": ml_response.get("objects", []),
                "mask": ml_response.get("mask"),
                "semanticMap": ml_response.get("semanticMap"),
                "metadata": {
                    **ml_response.get("metadata", {}),
                    "processingTime": time.time() - start_time,
                    "workerId": self.worker_id,
                },
            }

            # Mark as completed
            await self.update_job_status(
                job_id,
                "completed",
                progress=100,
                progress_message="Layout generation completed successfully!",
                response_data=response_data,
                completed_at=datetime.utcnow()
            )

            self.current_job_id = None
            return response_data

        except Exception as e:
            error_msg = f"Job processing failed: {str(e)}"
            await self.update_job_status(
                job_id,
                "failed",
                progress=0,
                error_message=error_msg,
                error_details={"exception": str(e)},
                completed_at=datetime.utcnow()
            )
            self.current_job_id = None
            raise

    async def update_job_status(
        self,
        job_id: int,
        status: str,
        progress: Optional[int] = None,
        progress_message: Optional[str] = None,
        response_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        error_details: Optional[dict] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ):
        """Update job status directly in database."""
        try:
            await self.db.update_layout_job(
                job_id=job_id,
                status=status,
                progress=progress,
                progress_message=progress_message,
                response_data=response_data,
                error_message=error_message,
                error_details=error_details,
                started_at=started_at,
                completed_at=completed_at,
                worker_id=self.worker_id,
            )
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")

    async def fetch_queued_jobs(self) -> list:
        """Fetch queued jobs directly from database."""
        try:
            jobs = await self.db.get_queued_jobs(limit=10)
            return jobs
        except Exception as e:
            print(f"Error fetching queued jobs: {e}")
            return []

    async def run(self, poll_interval: int = 3):
        """Run the worker, polling for new jobs."""
        self.running = True
        print(f"Worker {self.worker_id} started (ML Service: {self.ml_service_url}, polling every {poll_interval}s)")

        while self.running:
            try:
                # Fetch queued jobs
                queued_jobs = await self.fetch_queued_jobs()
                
                for job in queued_jobs:
                    if not self.running:
                        break
                    
                    job_id = job.get("id")
                    # Handle both camelCase and snake_case field names
                    request_data = job.get("request_data", {})
                    
                    if job_id and request_data:
                        print(f"[{self.worker_id}] Processing job {job_id}")
                        try:
                            await self.process_job(job_id, request_data)
                            print(f"[{self.worker_id}] Completed job {job_id}")
                        except Exception as e:
                            print(f"[{self.worker_id}] Error processing job {job_id}: {e}")
                
                await asyncio.sleep(poll_interval)
            except Exception as e:
                print(f"[{self.worker_id}] Error in worker loop: {e}")
                await asyncio.sleep(poll_interval)

    def stop(self):
        """Stop the worker."""
        self.running = False
        print(f"Worker {self.worker_id} stopped")


# Standalone worker script
async def main():
    """Main entry point for the stub worker."""
    worker = StubWorker()
    
    try:
        await worker.run(poll_interval=3)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
