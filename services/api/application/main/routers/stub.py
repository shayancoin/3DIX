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
    """
    List all stub projects stored in memory.
    
    Returns:
        List[StubProject]: The in-memory list of stub projects.
    """
    return _projects


@router.post("/", response_model=StubProject, status_code=201)
async def create_project(payload: CreateProject):
    """
    Create and store a new in-memory stub project with a generated id and timestamps.
    
    Returns:
        StubProject: The created project instance.
    """
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
    """
    Retrieve a stub project matching the given project identifier.
    
    Parameters:
        project_id (str): Project identifier to match against the project's `id` or MongoDB `_id` alias.
    
    Returns:
        StubProject: The matching stub project.
    
    Raises:
        HTTPException: With status code 404 if no project matches the provided identifier.
    """
    for proj in _projects:
        if proj.id == project_id or proj._id == project_id:
            return proj
    raise HTTPException(status_code=404, detail="Project not found")


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str):
    """
    Remove a stub project and all rooms associated with the given project identifier.
    
    This function removes any project whose `id` or `_id` equals `project_id` and also removes all rooms whose `project_id` equals `project_id` from the module's in-memory stores.
    
    Parameters:
        project_id (str): Project identifier to delete; matched against both the project's `id` and its `_id` alias.
    """
    global _projects, _rooms
    _projects = [p for p in _projects if p.id != project_id and p._id != project_id]
    _rooms = [r for r in _rooms if r.project_id != project_id]
    return None


rooms_router = APIRouter()


@rooms_router.get("/rooms", response_model=List[StubRoom])
async def list_rooms(project_id: Optional[str] = None):
    """
    List stub rooms, optionally filtered by project ID.
    
    Parameters:
        project_id (Optional[str]): If provided, only rooms whose `project_id` equals this value are returned.
    
    Returns:
        List[StubRoom]: Rooms matching the filter; all stub rooms if `project_id` is `None`.
    """
    if project_id:
        return [r for r in _rooms if r.project_id == project_id]
    return _rooms


@rooms_router.post("/rooms", response_model=StubRoom, status_code=201)
async def create_room(payload: CreateRoom):
    """
    Create and store a new in-memory room for the specified project.
    
    Parameters:
        payload (CreateRoom): The room data including project_id, name, type, width, length, and height.
    
    Returns:
        StubRoom: The created room with assigned `id`, `_id` (alias `mongo_id`), and timestamps.
    """
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
    """
    Retrieve a stub room by its identifier.
    
    Parameters:
        room_id (str): Identifier to match against the room's `id` or `_id`.
    
    Returns:
        StubRoom: The matching stub room.
    
    Raises:
        HTTPException: 404 if no room with the given identifier exists.
    """
    for room in _rooms:
        if room.id == room_id or room._id == room_id:
            return room
    raise HTTPException(status_code=404, detail="Room not found")


# In-memory stores (simple and sufficient for UI boot)
_projects: List[StubProject] = [StubProject()]
_rooms: List[StubRoom] = [StubRoom()]