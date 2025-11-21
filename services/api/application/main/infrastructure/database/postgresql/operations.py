"""
PostgreSQL database operations for 3DIX API.
Uses asyncpg for async PostgreSQL operations.
"""

import asyncpg
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC

from application.main.infrastructure.database.db_interface import DataBaseOperations


class PostgreSQL(DataBaseOperations, ABC):
    """PostgreSQL database operations using asyncpg."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        # asyncpg uses postgresql:// format, but we need to ensure it's correct
        conn_str = os.getenv(
            "POSTGRES_URL",
            os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/3dix")
        )
        # Ensure the connection string is in the correct format for asyncpg
        if conn_str.startswith("postgres://"):
            conn_str = conn_str.replace("postgres://", "postgresql://", 1)
        self.connection_string = conn_str

    async def initialize(self):
        """Initialize the connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60
            )

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def _get_pool(self):
        """Get or initialize the connection pool."""
        if not self.pool:
            await self.initialize()
        return self.pool

    # Generic database operations (for compatibility with existing interface)
    async def fetch_single_db_record(self, unique_id: str):
        """Generic fetch - not used for structured queries."""
        raise NotImplementedError("Use specific query methods instead")

    async def fetch_multiple_db_record(self, unique_id: str):
        """Generic fetch - not used for structured queries."""
        raise NotImplementedError("Use specific query methods instead")

    async def insert_single_db_record(self, record: Dict):
        """Generic insert - not used for structured queries."""
        raise NotImplementedError("Use specific query methods instead")

    async def insert_multiple_db_record(self, record: Dict):
        """Generic insert - not used for structured queries."""
        raise NotImplementedError("Use specific query methods instead")

    async def update_single_db_record(self, record: Dict):
        """Generic update - not used for structured queries."""
        raise NotImplementedError("Use specific query methods instead")

    async def update_multiple_db_record(self, record: Dict):
        """Generic update - not used for structured queries."""
        raise NotImplementedError("Use specific query methods instead")

    # Specific query methods for Projects
    async def get_projects_by_team(self, team_id: int) -> List[Dict[str, Any]]:
        """Get all projects for a team."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, team_id, name, description, created_at, updated_at, created_by, deleted_at
                FROM projects
                WHERE team_id = $1 AND deleted_at IS NULL
                ORDER BY created_at DESC
                """,
                team_id
            )
            return [dict(row) for row in rows]

    async def get_project_by_id(self, project_id: int, team_id: int) -> Optional[Dict[str, Any]]:
        """Get a project by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, team_id, name, description, created_at, updated_at, created_by, deleted_at
                FROM projects
                WHERE id = $1 AND team_id = $2 AND deleted_at IS NULL
                """,
                project_id, team_id
            )
            return dict(row) if row else None

    async def create_project(
        self,
        team_id: int,
        name: str,
        description: Optional[str] = None,
        created_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new project."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO projects (team_id, name, description, created_by, created_at, updated_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                RETURNING id, team_id, name, description, created_at, updated_at, created_by, deleted_at
                """,
                team_id, name, description, created_by
            )
            return dict(row)

    async def update_project(
        self,
        project_id: int,
        team_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a project."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            updates = []
            params = []
            param_idx = 1

            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1
            if description is not None:
                updates.append(f"description = ${param_idx}")
                params.append(description)
                param_idx += 1

            if not updates:
                return await self.get_project_by_id(project_id, team_id)

            updates.append(f"updated_at = NOW()")
            params.extend([project_id, team_id])

            row = await conn.fetchrow(
                f"""
                UPDATE projects
                SET {', '.join(updates)}
                WHERE id = ${param_idx} AND team_id = ${param_idx + 1} AND deleted_at IS NULL
                RETURNING id, team_id, name, description, created_at, updated_at, created_by, deleted_at
                """,
                *params
            )
            return dict(row) if row else None

    async def delete_project(self, project_id: int, team_id: int) -> bool:
        """Soft delete a project."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE projects
                SET deleted_at = NOW()
                WHERE id = $1 AND team_id = $2 AND deleted_at IS NULL
                """,
                project_id, team_id
            )
            return result == "UPDATE 1"

    # Specific query methods for Rooms
    async def get_rooms_by_project(self, project_id: int, team_id: int) -> List[Dict[str, Any]]:
        """Get all rooms for a project."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT r.id, r.project_id, r.name, r.room_type, r.width, r.height, r.length,
                       r.layout_data, r.scene_data, r.vibe_spec, r.thumbnail_url, r.is_active,
                       r.created_at, r.updated_at, r.created_by, r.deleted_at
                FROM rooms r
                INNER JOIN projects p ON r.project_id = p.id
                WHERE r.project_id = $1 AND p.team_id = $2 AND r.deleted_at IS NULL
                ORDER BY r.created_at DESC
                """,
                project_id, team_id
            )
            return [dict(row) for row in rows]

    async def get_room_by_id(self, room_id: int, project_id: int, team_id: int) -> Optional[Dict[str, Any]]:
        """Get a room by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT r.id, r.project_id, r.name, r.room_type, r.width, r.height, r.length,
                       r.layout_data, r.scene_data, r.vibe_spec, r.thumbnail_url, r.is_active,
                       r.created_at, r.updated_at, r.created_by, r.deleted_at
                FROM rooms r
                INNER JOIN projects p ON r.project_id = p.id
                WHERE r.id = $1 AND r.project_id = $2 AND p.team_id = $3 AND r.deleted_at IS NULL
                """,
                room_id, project_id, team_id
            )
            return dict(row) if row else None

    async def create_room(
        self,
        project_id: int,
        name: str,
        room_type: str,
        width: Optional[float] = None,
        height: Optional[float] = None,
        length: Optional[float] = None,
        created_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new room."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO rooms (project_id, name, room_type, width, height, length, created_by, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                RETURNING id, project_id, name, room_type, width, height, length,
                          layout_data, scene_data, vibe_spec, thumbnail_url, is_active,
                          created_at, updated_at, created_by, deleted_at
                """,
                project_id, name, room_type, width, height, length, created_by
            )
            return dict(row)

    async def update_room(
        self,
        room_id: int,
        project_id: int,
        team_id: int,
        name: Optional[str] = None,
        room_type: Optional[str] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        length: Optional[float] = None,
        layout_data: Optional[Dict] = None,
        scene_data: Optional[Dict] = None,
        vibe_spec: Optional[Dict] = None,
        thumbnail_url: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a room."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            updates = []
            params = []
            param_idx = 1

            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1
            if room_type is not None:
                updates.append(f"room_type = ${param_idx}")
                params.append(room_type)
                param_idx += 1
            if width is not None:
                updates.append(f"width = ${param_idx}")
                params.append(width)
                param_idx += 1
            if height is not None:
                updates.append(f"height = ${param_idx}")
                params.append(height)
                param_idx += 1
            if length is not None:
                updates.append(f"length = ${param_idx}")
                params.append(length)
                param_idx += 1
            if layout_data is not None:
                updates.append(f"layout_data = ${param_idx}")
                params.append(layout_data)
                param_idx += 1
            if scene_data is not None:
                updates.append(f"scene_data = ${param_idx}")
                params.append(scene_data)
                param_idx += 1
            if vibe_spec is not None:
                updates.append(f"vibe_spec = ${param_idx}")
                params.append(vibe_spec)
                param_idx += 1
            if thumbnail_url is not None:
                updates.append(f"thumbnail_url = ${param_idx}")
                params.append(thumbnail_url)
                param_idx += 1
            if is_active is not None:
                updates.append(f"is_active = ${param_idx}")
                params.append(is_active)
                param_idx += 1

            if not updates:
                return await self.get_room_by_id(room_id, project_id, team_id)

            updates.append("updated_at = NOW()")
            params.extend([room_id, project_id, team_id])

            row = await conn.fetchrow(
                f"""
                UPDATE rooms
                SET {', '.join(updates)}
                WHERE id = ${param_idx} AND project_id = ${param_idx + 1}
                  AND EXISTS (
                    SELECT 1 FROM projects p
                    WHERE p.id = rooms.project_id AND p.team_id = ${param_idx + 2} AND p.deleted_at IS NULL
                  )
                  AND deleted_at IS NULL
                RETURNING id, project_id, name, room_type, width, height, length,
                          layout_data, scene_data, vibe_spec, thumbnail_url, is_active,
                          created_at, updated_at, created_by, deleted_at
                """,
                *params
            )
            return dict(row) if row else None

    async def delete_room(self, room_id: int, project_id: int, team_id: int) -> bool:
        """Soft delete a room."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE rooms
                SET deleted_at = NOW()
                WHERE id = $1 AND project_id = $2
                  AND EXISTS (
                    SELECT 1 FROM projects p
                    WHERE p.id = rooms.project_id AND p.team_id = $3 AND p.deleted_at IS NULL
                  )
                  AND deleted_at IS NULL
                """,
                room_id, project_id, team_id
            )
            return result == "UPDATE 1"

    # Specific query methods for Layout Jobs
    async def create_layout_job(
        self,
        room_id: int,
        request_data: Dict[str, Any],
        status: str = "queued"
    ) -> Dict[str, Any]:
        """Create a new layout job."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO layout_jobs (room_id, status, request_data, progress, created_at, updated_at)
                VALUES ($1, $2, $3, 0, NOW(), NOW())
                RETURNING id, room_id, status, request_data, response_data, progress, progress_message,
                          error_message, error_details, created_at, started_at, completed_at, updated_at,
                          worker_id, retry_count
                """,
                room_id, status, request_data
            )
            return dict(row)

    async def get_job_by_id(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, room_id, status, request_data, response_data, progress, progress_message,
                       error_message, error_details, created_at, started_at, completed_at, updated_at,
                       worker_id, retry_count
                FROM layout_jobs
                WHERE id = $1
                """,
                job_id
            )
            return dict(row) if row else None

    async def get_queued_jobs(self) -> List[Dict[str, Any]]:
        """Get all queued jobs."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, room_id, status, request_data, response_data, progress, progress_message,
                       error_message, error_details, created_at, started_at, completed_at, updated_at,
                       worker_id, retry_count
                FROM layout_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 10
                """
            )
            return [dict(row) for row in rows]

    async def update_job(
        self,
        job_id: int,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        progress_message: Optional[str] = None,
        response_data: Optional[Dict] = None,
        error_message: Optional[str] = None,
        error_details: Optional[Dict] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        worker_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a job."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            updates = []
            params = []
            param_idx = 1

            if status is not None:
                updates.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1
            if progress is not None:
                updates.append(f"progress = ${param_idx}")
                params.append(progress)
                param_idx += 1
            if progress_message is not None:
                updates.append(f"progress_message = ${param_idx}")
                params.append(progress_message)
                param_idx += 1
            if response_data is not None:
                updates.append(f"response_data = ${param_idx}")
                params.append(response_data)
                param_idx += 1
            if error_message is not None:
                updates.append(f"error_message = ${param_idx}")
                params.append(error_message)
                param_idx += 1
            if error_details is not None:
                updates.append(f"error_details = ${param_idx}")
                params.append(error_details)
                param_idx += 1
            if started_at is not None:
                updates.append(f"started_at = ${param_idx}")
                params.append(started_at)
                param_idx += 1
            if completed_at is not None:
                updates.append(f"completed_at = ${param_idx}")
                params.append(completed_at)
                param_idx += 1
            if worker_id is not None:
                updates.append(f"worker_id = ${param_idx}")
                params.append(worker_id)
                param_idx += 1

            if not updates:
                return await self.get_job_by_id(job_id)

            updates.append("updated_at = NOW()")
            params.append(job_id)

            row = await conn.fetchrow(
                f"""
                UPDATE layout_jobs
                SET {', '.join(updates)}
                WHERE id = ${param_idx}
                RETURNING id, room_id, status, request_data, response_data, progress, progress_message,
                          error_message, error_details, created_at, started_at, completed_at, updated_at,
                          worker_id, retry_count
                """,
                *params
            )
            return dict(row) if row else None
