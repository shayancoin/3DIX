"""
API routes for custom object reconstruction using SAM-3D.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import httpx
import os

router = APIRouter()

SAM3D_SERVICE_URL = os.getenv("DEV_SAM3D_SERVICE_URL", os.getenv("SAM3D_SERVICE_URL", "http://localhost:8002"))


class ObjectReconstructionRequest(BaseModel):
    image: str  # base64 encoded image
    mask: Optional[str] = None  # base64 encoded mask
    mask_type: Optional[str] = "single"
    seed: Optional[int] = 42
    output_format: Optional[str] = "gltf"


class ObjectReconstructionResponse(BaseModel):
    object_id: str
    status: str
    mesh_url: Optional[str] = None
    mesh_data: Optional[str] = None
    format: str
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@router.post("/reconstruct", response_model=ObjectReconstructionResponse)
async def reconstruct_custom_object(request: ObjectReconstructionRequest):
    """
    Reconstruct a custom 3D object from an image using SAM-3D.
    This endpoint proxies to the SAM-3D service.
    """
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{SAM3D_SERVICE_URL}/reconstruct",
                json=request.dict(),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"SAM-3D service error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Check if SAM-3D service is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SAM3D_SERVICE_URL}/health")
            if response.status_code == 200:
                return {"status": "healthy", "sam3d_service": "available"}
            return {"status": "degraded", "sam3d_service": "unavailable"}
    except Exception:
        return {"status": "degraded", "sam3d_service": "unavailable"}
