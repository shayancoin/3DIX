from fastapi import APIRouter, HTTPException, Path, Body
from typing import List, Optional
from pydantic import BaseModel

from application.main.integrations.sam3d_objects import reconstruct_custom_object

router = APIRouter(prefix="/rooms/{room_id}/objects/{object_id}", tags=["custom-objects"])

class CustomMeshRequest(BaseModel):
    image_url: str
    mask: Optional[List[List[int]]] = None

class CustomMeshResponse(BaseModel):
    mesh_url: str
    preview_png_url: Optional[str] = None

@router.post("/custom-mesh", response_model=CustomMeshResponse)
async def create_custom_mesh(
    room_id: str = Path(...),
    object_id: str = Path(...),
    request: CustomMeshRequest = Body(...)
):
    # In a real implementation, we would:
    # 1. Verify user has access to room_id
    # 2. Fetch the object from DB to get its category and size
    # 3. Update the object in DB with the new mesh_url

    # For now, we'll mock the object data
    mock_category = "chair"
    mock_target_size = [0.5, 1.0, 0.5]

    try:
        result = await reconstruct_custom_object(
            image_url=request.image_url,
            category_hint=mock_category,
            target_size=mock_target_size,
            mask=request.mask
        )
        return CustomMeshResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
