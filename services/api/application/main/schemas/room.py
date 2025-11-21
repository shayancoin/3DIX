from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import uuid4

def generate_id():
    """
    Generate a new unique identifier string.
    
    Returns:
        id (str): A UUID4-based unique identifier.
    """
    return str(uuid4())

class RoomDimensions(BaseModel):
    width: float = Field(..., gt=0, description="Width in meters")
    length: float = Field(..., gt=0, description="Length in meters")
    height: float = Field(..., gt=0, description="Height in meters")

class RoomBase(BaseModel):
    name: str = Field(..., title="Room Name", min_length=1, max_length=100)
    type: str = Field(..., title="Room Type", description="e.g., living_room, bedroom")
    dimensions: RoomDimensions

class RoomCreate(RoomBase):
    pass

class RoomUpdate(BaseModel):
    name: Optional[str] = Field(None, title="Room Name")
    type: Optional[str] = Field(None, title="Room Type")
    dimensions: Optional[RoomDimensions] = None

class Room(RoomBase):
    id: str = Field(default_factory=generate_id, alias="_id")
    project_id: str = Field(..., title="Project ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True