from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional


class StubProject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # keep both "id" and "_id" so the frontend can consume either
    id: str = "proj-1"
    mongo_id: str = Field("proj-1", alias="_id")
    name: str = "Demo Project"
    description: Optional[str] = "Stub project"
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()


class StubRoom(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str = "room-1"
    mongo_id: str = Field("room-1", alias="_id")
    project_id: str = "proj-1"
    name: str = "Demo Room"
    type: str = "living_room"
    width: float = 5.0
    length: float = 4.0
    height: float = 2.5
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()


class CreateProject(BaseModel):
    name: str
    description: Optional[str] = None


class CreateRoom(BaseModel):
    project_id: str
    name: str
    type: str
    width: float
    length: float
    height: float


router = APIRouter()


@router.get("/", response_model=List[StubProject])
async def list_projects():
    """Return a stub list of projects."""
    return _projects


@router.post("/", response_model=StubProject, status_code=201)
async def create_project(payload: CreateProject):
    """Create a new stub project (in-memory only)."""
    new_id = f"proj-{len(_projects) + 1}"
    project = StubProject(
        id=new_id,
        mongo_id=new_id,
        name=payload.name,
        description=payload.description,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    _projects.append(project)
    return project


@router.get("/{project_id}", response_model=StubProject)
async def get_project(project_id: str):
    """Return a single stub project by id."""
    for proj in _projects:
        if proj.id == project_id or proj._id == project_id:
            return proj
    raise HTTPException(status_code=404, detail="Project not found")


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str):
    """Delete a stub project and its rooms."""
    global _projects, _rooms
    _projects = [p for p in _projects if p.id != project_id and p._id != project_id]
    _rooms = [r for r in _rooms if r.project_id != project_id]
    return None


rooms_router = APIRouter()


@rooms_router.get("/rooms", response_model=List[StubRoom])
async def list_rooms(project_id: Optional[str] = None):
    """Return a stub list of rooms (optionally filter by project_id)."""
    if project_id:
        return [r for r in _rooms if r.project_id == project_id]
    return _rooms


@rooms_router.post("/rooms", response_model=StubRoom, status_code=201)
async def create_room(payload: CreateRoom):
    """Create a stub room (in-memory only)."""
    new_id = f"room-{len(_rooms) + 1}"
    room = StubRoom(
        id=new_id,
        mongo_id=new_id,
        project_id=payload.project_id,
        name=payload.name,
        type=payload.type,
        width=payload.width,
        length=payload.length,
        height=payload.height,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    _rooms.append(room)
    return room


@rooms_router.get("/rooms/{room_id}", response_model=StubRoom)
async def get_room(room_id: str):
    """Return a stub room by id."""
    for room in _rooms:
        if room.id == room_id or room._id == room_id:
            return room
    raise HTTPException(status_code=404, detail="Room not found")


# In-memory stores (simple and sufficient for UI boot)
_projects: List[StubProject] = [StubProject()]
_rooms: List[StubRoom] = [StubRoom()]
