from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, status, Query
from application.main.schemas.room import Room, RoomCreate, RoomUpdate
from datetime import datetime

router = APIRouter(prefix="/rooms", tags=["rooms"])

@router.post("/", response_model=Room, status_code=status.HTTP_201_CREATED)
async def create_room(request: Request, room_in: RoomCreate):
    # Verify project exists
    project = await request.app.state.db_operations.find_one("projects", {"_id": room_in.project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    room = Room(**room_in.dict())
    room_dict = room.dict(by_alias=True)

    await request.app.state.db_operations.insert_single_db_record(room_dict, collection_name="rooms")
    return room

@router.get("/", response_model=List[Room])
async def list_rooms(request: Request, project_id: Optional[str] = Query(None)):
    query = {}
    if project_id:
        query["project_id"] = project_id

    rooms = await request.app.state.db_operations.find("rooms", query)
    return [Room(**r) for r in rooms]

@router.get("/{room_id}", response_model=Room)
async def get_room(request: Request, room_id: str):
    room_data = await request.app.state.db_operations.find_one("rooms", {"_id": room_id})
    if not room_data:
        raise HTTPException(status_code=404, detail="Room not found")
    return Room(**room_data)

@router.put("/{room_id}", response_model=Room)
async def update_room(request: Request, room_id: str, room_in: RoomUpdate):
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
    result = await request.app.state.db_operations.delete_one("rooms", {"_id": room_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")
    return None
