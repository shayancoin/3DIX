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
    Reconstruct a 3D mesh for an object in the given image using the SAM3D service.
    
    Parameters:
        image_url (str): URL of the source image containing the object.
        category_hint (str): Text hint for the object's category to guide reconstruction.
        target_size (List[float]): Desired object size as [width, height, depth] (units as expected by the service).
        mask (Optional[List[List[int]]]): Optional 2D mask array (e.g., binary values) indicating the object region in the image.
    
    Returns:
        Dict[str, Any]: Parsed JSON response from the SAM3D service describing the reconstructed mesh and metadata.
    
    Raises:
        httpx.HTTPStatusError: If the SAM3D service responds with a non-success HTTP status.
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