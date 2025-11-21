from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, status, Query
from application.main.schemas.room import Room, RoomCreate, RoomUpdate
from datetime import datetime

router = APIRouter(prefix="/rooms", tags=["rooms"])

@router.post("/", response_model=Room, status_code=status.HTTP_201_CREATED)
async def create_room(request: Request, room_in: RoomCreate):
    # Verify project exists
    """
    Create a new room after validating that the referenced project exists.
    
    Parameters:
        request (Request): FastAPI request carrying application state and DB operations.
        room_in (RoomCreate): Payload with room fields and the `project_id` to associate the room with.
    
    Returns:
        Room: The newly created Room model built from the input payload.
    
    Raises:
        HTTPException: 404 if the referenced project is not found.
    """
    project = await request.app.state.db_operations.find_one("projects", {"_id": room_in.project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    room = Room(**room_in.dict())
    room_dict = room.dict(by_alias=True)

    await request.app.state.db_operations.insert_single_db_record(room_dict, collection_name="rooms")
    return room

@router.get("/", response_model=List[Room])
async def list_rooms(request: Request, project_id: Optional[str] = Query(None)):
    """
    Retrieve rooms, optionally filtered by project ID.
    
    If `project_id` is provided, only rooms with that `project_id` are returned.
    
    Parameters:
    	project_id (Optional[str]): Project identifier to filter rooms; if omitted, all rooms are returned.
    
    Returns:
    	List[Room]: A list of Room objects matching the query.
    """
    query = {}
    if project_id:
        query["project_id"] = project_id

    rooms = await request.app.state.db_operations.find("rooms", query)
    return [Room(**r) for r in rooms]

@router.get("/{room_id}", response_model=Room)
async def get_room(request: Request, room_id: str):
    """
    Retrieve a room by its ID.
    
    Parameters:
        room_id (str): Identifier of the room to fetch.
    
    Returns:
        Room: The room reconstructed from the retrieved document.
    
    Raises:
        HTTPException: Raised with status code 404 if the room is not found.
    """
    room_data = await request.app.state.db_operations.find_one("rooms", {"_id": room_id})
    if not room_data:
        raise HTTPException(status_code=404, detail="Room not found")
    return Room(**room_data)

@router.put("/{room_id}", response_model=Room)
async def update_room(request: Request, room_id: str, room_in: RoomUpdate):
    """
    Update fields of an existing room.
    
    Only fields provided in `room_in` are persisted; when any update occurs the `updated_at` timestamp is set to the current UTC time.
    
    Parameters:
        room_in (RoomUpdate): Fields to apply to the room; only supplied fields will be persisted.
    
    Returns:
        Room: The room object after applying the requested updates.
    
    Raises:
        HTTPException: 404 if no room exists with the given `room_id`.
    """
    room_data = await request.app.state.db_operations.find_one("rooms", {"_id": room_id})
    if not room_data:
        raise HTTPException(status_code=404, detail="Room not found")

    update_data = room_in.dict(exclude_unset=True)
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await request.app.state.db_operations.update_one("rooms", {"_id": room_id}, {"$set": update_data})
        room_data.update(update_data)

    return Room(**room_data)

@router.delete("/{room_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_room(request: Request, room_id: str):
    """
    Delete a room by its ID.
    
    Parameters:
        room_id (str): Identifier of the room to delete.
    
    Raises:
        HTTPException: Raised with status code 404 if no room with the given ID exists.
    """
    result = await request.app.state.db_operations.delete_one("rooms", {"_id": room_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")
    return None