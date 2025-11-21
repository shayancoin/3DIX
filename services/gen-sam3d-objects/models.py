from pydantic import BaseModel
from typing import List, Optional

class ReconstructRequest(BaseModel):
    image_url: str
    mask: Optional[List[List[int]]] = None
    category_hint: str
    target_size: List[float]

class ReconstructResponse(BaseModel):
    mesh_url: str
    preview_png_url: Optional[str] = None
