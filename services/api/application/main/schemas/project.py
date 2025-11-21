from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4

def generate_id():
    return str(uuid4())

class ProjectBase(BaseModel):
    name: str = Field(..., title="Project Name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, title="Project Description")

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, title="Project Name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, title="Project Description")

class Project(ProjectBase):
    id: str = Field(default_factory=generate_id, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
