from fastapi import FastAPI, HTTPException
from models import ReconstructRequest, ReconstructResponse
import uvicorn

app = FastAPI(title="3DIX SAM-3D Objects Service")

@app.get("/health")
async def health():
    """
    Return basic service health information.
    
    Returns:
        dict: A dictionary with keys "status" and "service" where "status" is "ok" and "service" is the service identifier "gen-sam3d-objects".
    """
    return {"status": "ok", "service": "gen-sam3d-objects"}

@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(request: ReconstructRequest):
    # Stub implementation for now
    # In real implementation, this would call SAM-3D logic
    """
    Handle a reconstruction request and return placeholder asset URLs.
    
    Parameters:
        request (ReconstructRequest): Reconstruction input data (for example, contains fields like `category_hint`, images, and masks).
    
    Returns:
        ReconstructResponse: Response containing `mesh_url` (URL to a GLB mesh asset) and `preview_png_url` (URL to a preview PNG).
    """
    print(f"Received reconstruction request for {request.category_hint}")

    # Return a dummy mesh URL (e.g., a placeholder or a known asset)
    # For now, we'll return a placeholder URL
    return ReconstructResponse(
        mesh_url="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF-Binary/Duck.glb",
        preview_png_url="https://via.placeholder.com/150"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)