from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from application.main.infrastructure.database.postgresql.operations import PostgreSQL

router = APIRouter()


async def get_db() -> PostgreSQL:
    """Dependency to get PostgreSQL database instance."""
    db = PostgreSQL()
    await db.initialize()
    return db


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    team_id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

    class Config:
        from_attributes = True


class RoomCreate(BaseModel):
    name: str
    room_type: str
    width: Optional[float] = None
    height: Optional[float] = None
    length: Optional[float] = None


class RoomUpdate(BaseModel):
    name: Optional[str] = None
    room_type: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    length: Optional[float] = None
    layout_data: Optional[dict] = None
    scene_data: Optional[dict] = None
    vibe_spec: Optional[dict] = None
    thumbnail_url: Optional[str] = None
    is_active: Optional[bool] = None


class RoomResponse(BaseModel):
    id: int
    project_id: int
    name: str
    room_type: str
    width: Optional[float]
    height: Optional[float]
    length: Optional[float]
    layout_data: Optional[dict]
    scene_data: Optional[dict]
    vibe_spec: Optional[dict]
    thumbnail_url: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[int]

    class Config:
        from_attributes = True


@router.get("/", response_model=List[ProjectResponse])
async def get_projects(team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Get all projects for a team.
    """
    try:
        projects = await db.get_projects_by_team(team_id)
        return [ProjectResponse(**p) for p in projects]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(project: ProjectCreate, team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Create a new project.
    """
    try:
        created = await db.create_project(
            team_id=team_id,
            name=project.name,
            description=project.description,
            created_by=None  # TODO: Get from auth context
        )
        return ProjectResponse(**created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Get a specific project by ID.
    """
    try:
        project = await db.get_project_by_id(project_id, team_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectResponse(**project)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: int, team_id: int, project: ProjectUpdate, db: PostgreSQL = Depends(get_db)):
    """
    Update a project.
    """
    try:
        updated = await db.update_project(
            project_id=project_id,
            team_id=team_id,
            name=project.name,
            description=project.description
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectResponse(**updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: int, team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Delete a project (soft delete).
    """
    try:
        deleted = await db.delete_project(project_id, team_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Project not found")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/{project_id}/rooms", response_model=List[RoomResponse])
async def get_rooms(project_id: int, team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Get all rooms for a project.
    """
    try:
        rooms = await db.get_rooms_by_project(project_id, team_id)
        return [RoomResponse(**r) for r in rooms]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/{project_id}/rooms", response_model=RoomResponse, status_code=201)
async def create_room(project_id: int, team_id: int, room: RoomCreate, db: PostgreSQL = Depends(get_db)):
    """
    Create a new room in a project.
    """
    try:
        # Verify project exists and belongs to team
        project = await db.get_project_by_id(project_id, team_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        created = await db.create_room(
            project_id=project_id,
            name=room.name,
            room_type=room.room_type,
            width=room.width,
            height=room.height,
            length=room.length,
            created_by=None  # TODO: Get from auth context
        )
        return RoomResponse(**created)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/{project_id}/rooms/{room_id}", response_model=RoomResponse)
async def get_room(project_id: int, room_id: int, team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Get a specific room by ID.
    """
    try:
        room = await db.get_room_by_id(room_id, project_id, team_id)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        return RoomResponse(**room)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.patch("/{project_id}/rooms/{room_id}", response_model=RoomResponse)
async def update_room(project_id: int, room_id: int, team_id: int, room: RoomUpdate, db: PostgreSQL = Depends(get_db)):
    """
    Update a room.
    """
    try:
        updated = await db.update_room(
            room_id=room_id,
            project_id=project_id,
            team_id=team_id,
            name=room.name,
            room_type=room.room_type,
            width=room.width,
            height=room.height,
            length=room.length,
            layout_data=room.layout_data,
            scene_data=room.scene_data,
            vibe_spec=room.vibe_spec,
            thumbnail_url=room.thumbnail_url,
            is_active=room.is_active
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Room not found")
        return RoomResponse(**updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.delete("/{project_id}/rooms/{room_id}", status_code=204)
async def delete_room(project_id: int, room_id: int, team_id: int, db: PostgreSQL = Depends(get_db)):
    """
    Delete a room (soft delete).
    """
    try:
        deleted = await db.delete_room(room_id, project_id, team_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Room not found")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
