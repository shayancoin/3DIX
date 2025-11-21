"""
Background worker for processing layout generation jobs.
This is a stub implementation that simulates job processing.
"""

import asyncio
import time
import uuid
from typing import Optional
from datetime import datetime
import httpx


class JobWorker:
    """Stub worker that simulates layout generation job processing."""

    def __init__(self, api_base_url: str = "http://localhost:8000", worker_id: Optional[str] = None):
        self.api_base_url = api_base_url
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.running = False
        self.current_job_id: Optional[int] = None

    async def process_job(self, job_id: int, request_data: dict) -> dict:
        """
        Process a layout generation job.
        This is a stub that simulates processing with progress updates.
        """
        self.current_job_id = job_id

        # Update job status to running
        await self.update_job_status(job_id, "running", progress=0, progress_message="Starting layout generation...")
        await asyncio.sleep(0.5)

        # Simulate processing steps
        steps = [
            (10, "Loading room specifications..."),
            (25, "Processing vibe specification..."),
            (40, "Generating 2D layout..."),
            (60, "Optimizing object placement..."),
            (80, "Validating layout constraints..."),
            (95, "Finalizing layout..."),
        ]

        for progress, message in steps:
            await asyncio.sleep(1.0)  # Simulate processing time
            await self.update_job_status(job_id, "running", progress=progress, progress_message=message)

        # Simulate completion
        await asyncio.sleep(0.5)
        
        # Generate stub response data
        response_data = {
            "jobId": str(job_id),
            "status": "completed",
            "objects": [
                {
                    "id": "obj-1",
                    "category": "refrigerator",
                    "position": [1.0, 0.0, 0.5],
                    "size": [0.6, 0.6, 1.8],
                    "orientation": 0.0,
                },
                {
                    "id": "obj-2",
                    "category": "sink",
                    "position": [2.5, 0.0, 0.5],
                    "size": [0.6, 0.6, 0.3],
                    "orientation": 1.57,
                },
                {
                    "id": "obj-3",
                    "category": "cabinet",
                    "position": [3.5, 0.0, 0.5],
                    "size": [1.0, 0.6, 0.9],
                    "orientation": 0.0,
                },
            ],
            "metadata": {
                "processingTime": 6.0,
                "modelVersion": "stub-v1.0",
            },
        }

        await self.update_job_status(
            job_id,
            "completed",
            progress=100,
            progress_message="Layout generation completed successfully!",
            response_data=response_data,
            completed_at=datetime.utcnow().isoformat(),
        )

        self.current_job_id = None
        return response_data

    async def update_job_status(
        self,
        job_id: int,
        status: str,
        progress: Optional[int] = None,
        progress_message: Optional[str] = None,
        response_data: Optional[dict] = None,
        error_message: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        """Update job status via API."""
        update_data = {
            "status": status,
        }

        if progress is not None:
            update_data["progress"] = progress
        if progress_message:
            update_data["progressMessage"] = progress_message
        if response_data:
            update_data["responseData"] = response_data
        if error_message:
            update_data["errorMessage"] = error_message
        if started_at:
            update_data["startedAt"] = started_at
        if completed_at:
            update_data["completedAt"] = completed_at

        try:
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.api_base_url}/api/jobs/{job_id}",
                    json=update_data,
                    timeout=10.0,
                )
                if response.status_code != 200:
                    print(f"Failed to update job {job_id}: {response.status_code}")
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")

    async def fetch_queued_jobs(self) -> list:
        """Fetch queued jobs from API."""
        try:
            async with httpx.AsyncClient() as client:
                # Note: This endpoint would need to be implemented in the API
                # For now, we'll use a direct database query approach
                # In a real implementation, this would call an API endpoint
                response = await client.get(
                    f"{self.api_base_url}/api/v1/jobs/queued",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return response.json()
                return []
        except Exception as e:
            print(f"Error fetching queued jobs: {e}")
            return []

    async def run(self, poll_interval: int = 5):
        """Run the worker, polling for new jobs."""
        self.running = True
        print(f"Worker {self.worker_id} started")

        while self.running:
            try:
                # In a real implementation, fetch queued jobs from API
                # For now, we'll simulate by checking if there are any jobs
                # that need processing
                
                # This is a stub - in production, you would:
                # 1. Query the database for queued jobs
                # 2. Process each job
                # 3. Update job status
                
                await asyncio.sleep(poll_interval)
            except Exception as e:
                print(f"Error in worker loop: {e}")
                await asyncio.sleep(poll_interval)

    def stop(self):
        """Stop the worker."""
        self.running = False
        print(f"Worker {self.worker_id} stopped")


# Standalone worker script
async def main():
    """Main entry point for the worker."""
    import os
    
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    worker = JobWorker(api_base_url=api_url)
    
    try:
        await worker.run(poll_interval=5)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
