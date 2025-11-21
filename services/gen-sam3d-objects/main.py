"""
SAM-3D Object Reconstruction Service
Reconstructs 3D objects from images with masks using SAM-3D Objects model.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import sys
import base64
import io
from PIL import Image
import numpy as np

# Add research code to path
RESEARCH_PATH = os.path.join(os.path.dirname(__file__), "../../research/sam-3d-objects")
if RESEARCH_PATH not in sys.path:
    sys.path.insert(0, RESEARCH_PATH)

try:
    # Try to import SAM-3D inference
    sys.path.append(os.path.join(RESEARCH_PATH, "notebook"))
    from inference import Inference, load_image, load_single_mask
    SAM3D_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM-3D not available: {e}")
    SAM3D_AVAILABLE = False

app = FastAPI(
    title="3DIX SAM-3D Object Reconstruction Service",
    description="Service for reconstructing 3D objects from images using SAM-3D",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SAM-3D model
sam3d_inference = None
if SAM3D_AVAILABLE:
    try:
        checkpoint_tag = os.getenv("SAM3D_CHECKPOINT_TAG", "hf")
        config_path = os.path.join(RESEARCH_PATH, f"checkpoints/{checkpoint_tag}/pipeline.yaml")
        if os.path.exists(config_path):
            sam3d_inference = Inference(config_path, compile=False)
            print(f"SAM-3D model loaded from {config_path}")
        else:
            print(f"SAM-3D config not found at {config_path}, using stub mode")
    except Exception as e:
        print(f"Warning: Failed to initialize SAM-3D: {e}")
        print("Falling back to stub mode")


# Request/Response Models

class ObjectReconstructionRequest(BaseModel):
    image: str  # base64 encoded image
    mask: Optional[str] = None  # base64 encoded mask image
    mask_type: Optional[str] = "single"  # 'single' or 'multi'
    seed: Optional[int] = 42
    output_format: Optional[str] = "gltf"  # 'gltf', 'ply', 'obj'


class ObjectReconstructionResponse(BaseModel):
    object_id: str
    status: str  # 'completed' | 'failed'
    mesh_url: Optional[str] = None  # URL to mesh file
    mesh_data: Optional[str] = None  # base64 encoded mesh (for small meshes)
    format: str  # 'gltf' | 'ply' | 'obj'
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gen-sam3d-objects",
        "model_available": SAM3D_AVAILABLE and sam3d_inference is not None
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "3DIX SAM-3D Object Reconstruction Service",
        "version": "0.1.0",
        "status": "stub" if not sam3d_inference else "ready"
    }


@app.post("/reconstruct", response_model=ObjectReconstructionResponse)
async def reconstruct_object(request: ObjectReconstructionRequest):
    """
    Reconstruct a 3D object from an image and optional mask.
    
    If mask is not provided, the entire image will be used.
    """
    try:
        # Decode image
        image_data = request.image.split(",")[1] if "," in request.image else request.image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)

        # Decode mask if provided
        mask_array = None
        if request.mask:
            mask_data = request.mask.split(",")[1] if "," in request.mask else request.mask
            mask_bytes = base64.b64decode(mask_data)
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask_array = np.array(mask_image)

        # Run reconstruction
        if sam3d_inference and SAM3D_AVAILABLE:
            try:
                output = sam3d_inference(
                    image_array,
                    mask_array,
                    seed=request.seed
                )
                
                # Export mesh based on format
                mesh_data = None
                if request.output_format == "ply":
                    # Export as PLY
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                        output["gs"].save_ply(tmp.name)
                        with open(tmp.name, "rb") as f:
                            mesh_data = base64.b64encode(f.read()).decode()
                        os.unlink(tmp.name)
                else:
                    # Default: return gaussian splat data or convert to mesh
                    # For now, return PLY format
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                        output["gs"].save_ply(tmp.name)
                        with open(tmp.name, "rb") as f:
                            mesh_data = base64.b64encode(f.read()).decode()
                        os.unlink(tmp.name)

                return ObjectReconstructionResponse(
                    object_id=f"obj-{request.seed}",
                    status="completed",
                    mesh_data=mesh_data,
                    format=request.output_format or "ply",
                    metadata={
                        "model": "sam3d",
                        "seed": request.seed,
                    }
                )
            except Exception as e:
                return ObjectReconstructionResponse(
                    object_id=f"obj-{request.seed}",
                    status="failed",
                    error_message=str(e),
                    format=request.output_format or "ply"
                )
        else:
            # Stub response
            return ObjectReconstructionResponse(
                object_id=f"stub-obj-{request.seed}",
                status="completed",
                mesh_data=None,  # Stub doesn't generate actual mesh
                format=request.output_format or "gltf",
                metadata={
                    "model": "stub",
                    "seed": request.seed,
                    "note": "SAM-3D model not available, returning stub response"
                }
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")


@app.post("/reconstruct/multi", response_model=List[ObjectReconstructionResponse])
async def reconstruct_multiple_objects(
    image: UploadFile = File(...),
    masks: List[UploadFile] = File(...),
    seed: int = 42,
    output_format: str = "gltf"
):
    """
    Reconstruct multiple 3D objects from a single image with multiple masks.
    """
    try:
        # Read image
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image_pil)

        results = []
        
        for i, mask_file in enumerate(masks):
            mask_bytes = await mask_file.read()
            mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask_array = np.array(mask_pil)

            if sam3d_inference and SAM3D_AVAILABLE:
                try:
                    output = sam3d_inference(image_array, mask_array, seed=seed + i)
                    
                    # Export mesh
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                        output["gs"].save_ply(tmp.name)
                        with open(tmp.name, "rb") as f:
                            mesh_data = base64.b64encode(f.read()).decode()
                        os.unlink(tmp.name)

                    results.append(ObjectReconstructionResponse(
                        object_id=f"obj-{i}",
                        status="completed",
                        mesh_data=mesh_data,
                        format=output_format,
                        metadata={"model": "sam3d", "seed": seed + i}
                    ))
                except Exception as e:
                    results.append(ObjectReconstructionResponse(
                        object_id=f"obj-{i}",
                        status="failed",
                        error_message=str(e),
                        format=output_format
                    ))
            else:
                # Stub response
                results.append(ObjectReconstructionResponse(
                    object_id=f"stub-obj-{i}",
                    status="completed",
                    format=output_format,
                    metadata={"model": "stub", "seed": seed + i}
                ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-object reconstruction failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
