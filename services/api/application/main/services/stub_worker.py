"""
Stub worker for Step 3 that generates dummy layouts without calling ML service.
This will be replaced in Step 4 with real ML service integration.
"""

import asyncio
import time
import uuid
from typing import Optional
from datetime import datetime
import os
from application.main.infrastructure.database.postgresql.operations import Postgresql


class StubWorker:
    """Stub worker that generates dummy layout results for Step 3."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        db: Optional[Postgresql] = None
    ):
        self.db = db or Postgresql()
        self.worker_id = worker_id or f"stub-worker-{uuid.uuid4().hex[:8]}"
        self.running = False
        self.current_job_id: Optional[int] = None

    def generate_dummy_layout(self, request_data: dict) -> dict:
        """
        Generate a dummy layout response for Step 3.
        This creates a simple layout with a few objects.
        """
        # Extract room type from request data (can be nested in vibeSpec or constraints)
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
            "objects": objects,
            "world_scale": 0.01,
            "semantic_map_png_url": None,  # Will be added in Step 5
            "room_outline": None,  # Will be added in Step 5
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

            # Simulate processing time (3-5 seconds)
            await asyncio.sleep(1)
            await self.update_job_status(
                job_id,
                "running",
                progress=30,
                progress_message="Generating layout..."
            )

            await asyncio.sleep(1)
            await self.update_job_status(
                job_id,
                "running",
                progress=60,
                progress_message="Processing objects..."
            )

            await asyncio.sleep(1)
            await self.update_job_status(
                job_id,
                "running",
                progress=90,
                progress_message="Finalizing layout..."
            )

            # Generate dummy layout
            response_data = self.generate_dummy_layout(request_data)
            response_data["metadata"] = {
                "processingTime": time.time() - start_time,
                "workerId": self.worker_id,
                "stub": True,  # Mark as stub for Step 3
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
        print(f"Stub Worker {self.worker_id} started (polling every {poll_interval}s)")

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
        print(f"Stub Worker {self.worker_id} stopped")


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
