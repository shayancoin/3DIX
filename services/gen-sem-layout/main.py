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
    
    This is a stub implementation that simulates layout generation.
    In Step 5, this will be replaced with real SemLayoutDiff inference.
    """
    start_time = time.time()
    
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    # Generate stub layout objects based on room type
    room_type = request.vibeSpec.prompt.roomType
    room_dims = request.constraints.roomDimensions if request.constraints else None
    
    # Default room dimensions if not provided
    width = room_dims.width if room_dims else 5.0
    length = room_dims.length if room_dims else 4.0
    
    # Generate objects based on room type
    objects = generate_stub_objects(room_type, width, length)
    
    processing_time = time.time() - start_time
    
    return LayoutResponse(
        jobId=request.roomId,  # Using roomId as jobId for stub
        status="completed",
        objects=objects,
        metadata={
            "processingTime": processing_time,
            "modelVersion": "stub-v1.0",
            "roomType": room_type,
        }
    )


def generate_stub_objects(room_type: str, width: float, length: float) -> List[LayoutObject]:
    """Generate stub layout objects based on room type."""
    objects = []
    
    if room_type == "kitchen":
        objects = [
            LayoutObject(
                id="obj-1",
                category="refrigerator",
                position=[0.5, 0.0, 0.3],
                size=[0.6, 1.8, 0.6],
                orientation=0.0,
            ),
            LayoutObject(
                id="obj-2",
                category="sink",
                position=[width * 0.4, 0.0, length * 0.5],
                size=[0.6, 0.3, 0.6],
                orientation=1.57,
            ),
            LayoutObject(
                id="obj-3",
                category="stove",
                position=[width * 0.7, 0.0, length * 0.5],
                size=[0.6, 0.3, 0.6],
                orientation=1.57,
            ),
            LayoutObject(
                id="obj-4",
                category="cabinet",
                position=[width * 0.2, 0.0, length * 0.3],
                size=[1.0, 0.9, 0.6],
                orientation=0.0,
            ),
        ]
    elif room_type == "bathroom":
        objects = [
            LayoutObject(
                id="obj-1",
                category="toilet",
                position=[width * 0.3, 0.0, length * 0.2],
                size=[0.4, 0.4, 0.7],
                orientation=1.57,
            ),
            LayoutObject(
                id="obj-2",
                category="sink",
                position=[width * 0.7, 0.0, length * 0.3],
                size=[0.5, 0.3, 0.5],
                orientation=0.0,
            ),
            LayoutObject(
                id="obj-3",
                category="shower",
                position=[width * 0.2, 0.0, length * 0.7],
                size=[0.8, 2.0, 0.8],
                orientation=0.0,
            ),
        ]
    elif room_type == "bedroom":
        objects = [
            LayoutObject(
                id="obj-1",
                category="bed",
                position=[width * 0.5, 0.0, length * 0.4],
                size=[2.0, 0.5, 1.8],
                orientation=0.0,
            ),
            LayoutObject(
                id="obj-2",
                category="dresser",
                position=[width * 0.2, 0.0, length * 0.2],
                size=[1.2, 1.0, 0.5],
                orientation=1.57,
            ),
            LayoutObject(
                id="obj-3",
                category="nightstand",
                position=[width * 0.8, 0.0, length * 0.3],
                size=[0.5, 0.5, 0.5],
                orientation=0.0,
            ),
        ]
    else:
        # Default generic objects
        objects = [
            LayoutObject(
                id="obj-1",
                category="furniture",
                position=[width * 0.3, 0.0, length * 0.3],
                size=[1.0, 0.5, 1.0],
                orientation=0.0,
            ),
            LayoutObject(
                id="obj-2",
                category="furniture",
                position=[width * 0.7, 0.0, length * 0.7],
                size=[1.0, 0.5, 1.0],
                orientation=1.57,
            ),
        ]
    
    return objects


if __name__ == "__main__":
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8001)
