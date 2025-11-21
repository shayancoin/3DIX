"""
PostgreSQL database operations using asyncpg.
Implements the DataBaseOperations interface for PostgreSQL.
"""
import asyncpg
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from application.main.infrastructure.database.db_interface import DataBaseOperations
from application.main.config import settings
import os


class Postgresql(DataBaseOperations):
    """PostgreSQL database operations implementation."""

    def __init__(self):
        super().__init__()
        self._pool: Optional[asyncpg.Pool] = None
        self._connection_string = os.getenv('POSTGRES_URL') or os.getenv('DEV_POSTGRES_URL')

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            if not self._connection_string:
                raise ValueError("POSTGRES_URL environment variable is not set")
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # Generic database operations (for compatibility with interface)
    async def fetch_single_db_record(self, unique_id: str):
        """Generic fetch - not used for Projects/Rooms."""
        raise NotImplementedError("Use specific methods for Projects/Rooms")

    async def update_single_db_record(self, record: Dict):
        """Generic update - not used for Projects/Rooms."""
        raise NotImplementedError("Use specific methods for Projects/Rooms")

    async def update_multiple_db_record(self, record: Dict):
        """Generic update - not used for Projects/Rooms."""
        raise NotImplementedError("Use specific methods for Projects/Rooms")

    async def fetch_multiple_db_record(self, unique_id: str):
        """Generic fetch - not used for Projects/Rooms."""
        raise NotImplementedError("Use specific methods for Projects/Rooms")

    async def insert_single_db_record(self, record: Dict):
        """Generic insert - not used for Projects/Rooms."""
        raise NotImplementedError("Use specific methods for Projects/Rooms")

    async def insert_multiple_db_record(self, record: Dict):
        """Generic insert - not used for Projects/Rooms."""
        raise NotImplementedError("Use specific methods for Projects/Rooms")

    # Projects operations
    async def get_projects(self, team_id: int) -> List[Dict[str, Any]]:
        """Get all projects for a team."""
        pool = await self.get_pool()
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

    async def get_project(self, project_id: int, team_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific project by ID."""
        pool = await self.get_pool()
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO projects (team_id, name, description, created_by)
                VALUES ($1, $2, $3, $4)
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Build update query dynamically
            updates = []
            values = []
            param_num = 1

            if name is not None:
                updates.append(f"name = ${param_num}")
                values.append(name)
                param_num += 1

            if description is not None:
                updates.append(f"description = ${param_num}")
                values.append(description)
                param_num += 1

            if not updates:
                # No updates, just return the existing project
                return await self.get_project(project_id, team_id)

            updates.append(f"updated_at = ${param_num}")
            values.append(datetime.utcnow())
            param_num += 1

            values.extend([project_id, team_id])

            row = await conn.fetchrow(
                f"""
                UPDATE projects
                SET {', '.join(updates)}
                WHERE id = ${param_num} AND team_id = ${param_num + 1} AND deleted_at IS NULL
                RETURNING id, team_id, name, description, created_at, updated_at, created_by, deleted_at
                """,
                *values
            )
            return dict(row) if row else None

    async def delete_project(self, project_id: int, team_id: int) -> bool:
        """Soft delete a project."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE projects
                SET deleted_at = $1
                WHERE id = $2 AND team_id = $3 AND deleted_at IS NULL
                """,
                datetime.utcnow(), project_id, team_id
            )
            return result == "UPDATE 1"

    # Rooms operations
    async def get_rooms(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all rooms for a project."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, project_id, name, room_type, width, height, length,
                       layout_data, scene_data, vibe_spec, thumbnail_url, is_active,
                       created_at, updated_at, created_by, deleted_at
                FROM rooms
                WHERE project_id = $1 AND deleted_at IS NULL
                ORDER BY created_at DESC
                """,
                project_id
            )
            # asyncpg returns JSONB fields as dict/list automatically
            return [dict(row) for row in rows]

    async def get_room(self, room_id: int, project_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific room by ID."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, project_id, name, room_type, width, height, length,
                       layout_data, scene_data, vibe_spec, thumbnail_url, is_active,
                       created_at, updated_at, created_by, deleted_at
                FROM rooms
                WHERE id = $1 AND project_id = $2 AND deleted_at IS NULL
                """,
                room_id, project_id
            )
            # asyncpg returns JSONB fields as dict/list automatically
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO rooms (project_id, name, room_type, width, height, length, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Build update query dynamically
            updates = []
            values = []
            param_num = 1

            if name is not None:
                updates.append(f"name = ${param_num}")
                values.append(name)
                param_num += 1

            if room_type is not None:
                updates.append(f"room_type = ${param_num}")
                values.append(room_type)
                param_num += 1

            if width is not None:
                updates.append(f"width = ${param_num}")
                values.append(width)
                param_num += 1

            if height is not None:
                updates.append(f"height = ${param_num}")
                values.append(height)
                param_num += 1

            if length is not None:
                updates.append(f"length = ${param_num}")
                values.append(length)
                param_num += 1

            if layout_data is not None:
                updates.append(f"layout_data = ${param_num}")
                values.append(json.dumps(layout_data))
                param_num += 1

            if scene_data is not None:
                updates.append(f"scene_data = ${param_num}")
                values.append(json.dumps(scene_data))
                param_num += 1

            if vibe_spec is not None:
                updates.append(f"vibe_spec = ${param_num}")
                values.append(json.dumps(vibe_spec))
                param_num += 1

            if thumbnail_url is not None:
                updates.append(f"thumbnail_url = ${param_num}")
                values.append(thumbnail_url)
                param_num += 1

            if is_active is not None:
                updates.append(f"is_active = ${param_num}")
                values.append(is_active)
                param_num += 1

            if not updates:
                # No updates, just return the existing room
                return await self.get_room(room_id, project_id)

            updates.append(f"updated_at = ${param_num}")
            values.append(datetime.utcnow())
            param_num += 1

            values.extend([room_id, project_id])

            row = await conn.fetchrow(
                f"""
                UPDATE rooms
                SET {', '.join(updates)}
                WHERE id = ${param_num} AND project_id = ${param_num + 1} AND deleted_at IS NULL
                RETURNING id, project_id, name, room_type, width, height, length,
                          layout_data, scene_data, vibe_spec, thumbnail_url, is_active,
                          created_at, updated_at, created_by, deleted_at
                """,
                *values
            )
            if row:
                result = dict(row)
                # asyncpg returns JSONB as dict/list, no parsing needed
                return result
            return None

    async def delete_room(self, room_id: int, project_id: int) -> bool:
        """Soft delete a room."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE rooms
                SET deleted_at = $1
                WHERE id = $2 AND project_id = $3 AND deleted_at IS NULL
                """,
                datetime.utcnow(), room_id, project_id
            )
            return result == "UPDATE 1"
