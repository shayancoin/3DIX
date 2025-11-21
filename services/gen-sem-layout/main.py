"""
ML Microservice for Semantic Layout Generation using SemLayoutDiff.
This is a stub implementation that will be replaced with real model inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import time
import os
from semlayoutdiff_integration import SemLayoutDiffIntegration

app = FastAPI(
    title="3DIX Layout Generation Service",
    description="ML service for generating semantic room layouts",
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

# Initialize SemLayoutDiff integration
sldn_checkpoint = os.getenv("SLDN_CHECKPOINT_PATH")
apm_checkpoint = os.getenv("APM_CHECKPOINT_PATH")
config_path = os.getenv("SEMLAYOUTDIFF_CONFIG_PATH")

semlayoutdiff = SemLayoutDiffIntegration(
    sldn_checkpoint_path=sldn_checkpoint,
    apm_checkpoint_path=apm_checkpoint,
    config_path=config_path
)


# Request/Response Models matching TypeScript types

class Point2D(BaseModel):
    x: float
    y: float


class VibePrompt(BaseModel):
    text: str
    referenceImageUrl: Optional[str] = None
    roomType: str


class VibeTag(BaseModel):
    id: str
    label: str
    category: str  # 'style' | 'mood' | 'color' | 'material' | 'era'
    weight: Optional[float] = 0.5


class VibeSlider(BaseModel):
    id: str
    label: str
    min: float
    max: float
    value: float
    step: Optional[float] = None


class VibeSpec(BaseModel):
    prompt: VibePrompt
    tags: List[VibeTag] = []
    sliders: List[VibeSlider] = []
    metadata: Optional[Dict[str, Any]] = None


class RoomDimensions(BaseModel):
    width: float
    height: float
    length: float


class SceneObject2D(BaseModel):
    id: str
    category: str
    label: Optional[str] = None
    position: Point2D
    size: Dict[str, float]  # { width: float, height: float }
    rotation: Optional[float] = None
    boundingBox: Dict[str, float]
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LayoutConstraints(BaseModel):
    roomDimensions: Optional[RoomDimensions] = None
    existingObjects: Optional[List[SceneObject2D]] = None
    maskType: Optional[str] = None  # 'none' | 'room_boundary' | 'wall_mask' | 'door_window_mask'
    maskImage: Optional[str] = None  # base64 encoded image


class LayoutRequest(BaseModel):
    roomId: str
    vibeSpec: VibeSpec
    constraints: Optional[LayoutConstraints] = None


class LayoutObject(BaseModel):
    id: str
    category: str
    position: List[float]  # [x, y, z]
    size: List[float]  # [width, height, depth]
    orientation: float  # radians
    metadata: Optional[Dict[str, Any]] = None


class LayoutResponse(BaseModel):
    jobId: str
    status: str  # 'queued' | 'running' | 'completed' | 'failed'
    mask: Optional[str] = None  # base64 or url
    objects: List[LayoutObject]
    semanticMap: Optional[str] = None  # base64 encoded semantic segmentation map
    metadata: Optional[Dict[str, Any]] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "gen-sem-layout"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "3DIX Layout Generation Service",
        "version": "0.1.0",
        "status": "stub"
    }


@app.post("/generate", response_model=LayoutResponse)
async def generate_layout(request: LayoutRequest):
    """
    Generate a semantic room layout based on vibe specification.
    
    This uses SemLayoutDiff for real layout generation when models are available,
    otherwise falls back to stub generation.
    """
    start_time = time.time()
    
    room_type = request.vibeSpec.prompt.roomType
    room_dims = request.constraints.roomDimensions if request.constraints else None
    
    # Default room dimensions if not provided
    width = room_dims.width if room_dims else 5.0
    length = room_dims.length if room_dims else 4.0
    
    # Extract floor plan mask if provided
    floor_plan_mask = None
    if request.constraints and request.constraints.maskImage:
        # Decode base64 mask image
        import base64
        from PIL import Image
        import io
        import numpy as np
        
        try:
            mask_data = request.constraints.maskImage.split(',')[1] if ',' in request.constraints.maskImage else request.constraints.maskImage
            mask_bytes = base64.b64decode(mask_data)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
            floor_plan_mask = np.array(mask_img)
        except Exception as e:
            print(f"Warning: Failed to decode mask image: {e}")
    
    # Generate semantic layout using SemLayoutDiff
    semantic_map, layout_metadata = semlayoutdiff.generate_semantic_layout(
        room_type=room_type,
        floor_plan_mask=floor_plan_mask,
        num_samples=1
    )
    
    # Predict 3D attributes from semantic map
    attribute_predictions = semlayoutdiff.predict_attributes(
        semantic_map=semantic_map,
        room_type=room_type
    )
    
    # Convert predictions to LayoutObject format
    objects = []
    for i, pred in enumerate(attribute_predictions):
        objects.append(
            LayoutObject(
                id=f"obj-{i+1}",
                category=pred["category"],
                position=pred["position"],
                size=pred["size"],
                orientation=pred["orientation"],
                metadata={
                    "source": "semlayoutdiff" if semlayoutdiff.initialized else "stub"
                }
            )
        )
    
    # Convert semantic map to base64 for response
    semantic_map_b64 = semlayoutdiff.semantic_map_to_base64(semantic_map)
    
    processing_time = time.time() - start_time
    
    return LayoutResponse(
        jobId=request.roomId,
        status="completed",
        mask=semantic_map_b64,
        semanticMap=semantic_map_b64,
        objects=objects,
        metadata={
            "processingTime": processing_time,
            "modelVersion": "semlayoutdiff-v1.0" if semlayoutdiff.initialized else "stub-v1.0",
            "roomType": room_type,
            "layoutMetadata": layout_metadata,
        }
    )




if __name__ == "__main__":
    import asyncio
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
