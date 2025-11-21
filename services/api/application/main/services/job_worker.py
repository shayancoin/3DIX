"""
Background worker for processing layout generation jobs.
This integrates with the ML microservice for layout generation.
"""

import asyncio
import time
import uuid
import os
from typing import Optional
from datetime import datetime
import httpx


class JobWorker:
    """Stub worker that simulates layout generation job processing."""

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        ml_service_url: str = None,
        worker_id: Optional[str] = None
    ):
        self.api_base_url = api_base_url
        self.ml_service_url = ml_service_url or os.getenv(
            "ML_SERVICE_URL",
            "http://localhost:8001"
        )
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.running = False
        self.current_job_id: Optional[int] = None

    async def process_job(self, job_id: int, request_data: dict) -> dict:
        """
        Process a layout generation job by calling the ML service.
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
                started_at=datetime.utcnow().isoformat()
            )

            # Call ML service
            await self.update_job_status(
                job_id,
                "running",
                progress=20,
                progress_message="Calling ML service..."
            )

            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    response = await client.post(
                        f"{self.ml_service_url}/generate",
                        json=request_data,
                    )
                    response.raise_for_status()
                    ml_response = response.json()
                except httpx.HTTPError as e:
                    error_msg = f"ML service error: {str(e)}"
                    await self.update_job_status(
                        job_id,
                        "failed",
                        progress=0,
                        error_message=error_msg,
                        completed_at=datetime.utcnow().isoformat(),
                    )
                    raise

            # Update progress
            await self.update_job_status(
                job_id,
                "running",
                progress=90,
                progress_message="Processing ML response..."
            )

            # Format response data
            response_data = {
                "jobId": str(job_id),
                "status": ml_response.get("status", "completed"),
                "objects": ml_response.get("objects", []),
                "mask": ml_response.get("mask"),
                "semanticMap": ml_response.get("semanticMap"),
                "metadata": {
                    **ml_response.get("metadata", {}),
                    "processingTime": time.time() - start_time,
                },
            }

            # Mark as completed
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

        except Exception as e:
            error_msg = f"Job processing failed: {str(e)}"
            await self.update_job_status(
                job_id,
                "failed",
                progress=0,
                error_message=error_msg,
                error_details={"exception": str(e)},
                completed_at=datetime.utcnow().isoformat(),
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
                response = await client.get(
                    f"{self.api_base_url}/api/v1/internal/jobs/queued",
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
        print(f"Worker {self.worker_id} started (ML Service: {self.ml_service_url})")

        while self.running:
            try:
                # Fetch queued jobs
                queued_jobs = await self.fetch_queued_jobs()
                
                for job in queued_jobs:
                    if not self.running:
                        break
                    
                    job_id = job.get("id")
                    request_data = job.get("request_data", {})
                    
                    if job_id and request_data:
                        print(f"Processing job {job_id}")
                        try:
                            await self.process_job(job_id, request_data)
                        except Exception as e:
                            print(f"Error processing job {job_id}: {e}")
                
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
