import httpx
from typing import List, Optional, Dict, Any

SAM3D_SERVICE_URL = "http://localhost:8004"

async def reconstruct_custom_object(
    image_url: str,
    category_hint: str,
    target_size: List[float],
    mask: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """
    Call the gen-sam3d-objects service to reconstruct a 3D mesh from an image.
    """
    payload = {
        "image_url": image_url,
        "mask": mask,
        "category_hint": category_hint,
        "target_size": target_size
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{SAM3D_SERVICE_URL}/reconstruct", json=payload)
        response.raise_for_status()
        return response.json()
