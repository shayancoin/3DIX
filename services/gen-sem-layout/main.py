from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Tuple
import os
import uvicorn
import numpy as np
import requests
import requests
from sem_layout_inference import generate_semantic_layout, to_layout_response, semantic_to_png_url
from vibe_encoder import VibeEncoder
from constraint_solver import ConstraintSolver, LayoutObject as ConstraintLayoutObject
from room_configs import get_room_type_config

app = FastAPI(
    title="3DIX Layout Generation Service",
    description="Stub layout generator (SemLayoutDiff placeholder)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vibe Encoder
vibe_encoder = VibeEncoder()


# ---------------------------
# Models (mirror packages/types contract)
# ---------------------------

class VibeSpec(BaseModel):
    prompt: dict
    tags: List[dict] = []
    sliders: List[dict] = []
    metadata: Optional[dict] = None


class LayoutRequest(BaseModel):
    room_type: str
    arch_mask_url: Optional[str] = None
    mask_type: Optional[str] = "none"
    vibe_spec: VibeSpec
    seed: Optional[int] = None


class SceneObject3D(BaseModel):
    id: str
    category: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    orientation: int  # 0|1|2|3
    metadata: Optional[dict] = None


class LayoutResponse(BaseModel):
    semantic_map_png_url: Optional[str] = None
    objects: List[SceneObject3D]
    world_scale: float
    room_outline: Optional[List[Tuple[float, float]]] = None


# ---------------------------
# Health
# ---------------------------

@app.get("/health")
async def health():
    """
    Report the service health.

    Returns:
        dict: A mapping with keys "status" (value "ok") and "service" (value "gen-sem-layout").
    """
    return {"status": "ok", "service": "gen-sem-layout"}


@app.get("/")
async def root():
    """
    Return basic service identity and status for the root HTTP endpoint.

    Returns:
        dict: A mapping with keys:
            - "service": the service name ("gen-sem-layout"),
            - "version": the API version string ("0.2.0"),
            - "status": a short status indicator ("stub").
    """
    return {"service": "gen-sem-layout", "version": "0.2.0", "status": "stub"}


# ---------------------------
# Stub generation
# ---------------------------

def stub_objects(seed: int) -> List[SceneObject3D]:
    """
    Generate a deterministic list of three fallback SceneObject3D objects for a given seed.

    Parameters:
        seed (int): Integer seed used to deterministically vary randomized attributes (affects the chair's position and orientation).

    Returns:
        List[SceneObject3D]: A list containing three SceneObject3D instances (sofa, table, chair). Each object's `metadata` includes the provided `seed` and the generator identifier `"fallback-stub"`.
    """
    import random

    random.seed(seed)
    base = [
        SceneObject3D(
            id="sofa-1",
            category="sofa",
            position=(1.0, 0.0, 2.0),
            size=(2.0, 1.0, 1.0),
            orientation=1,
            metadata={"seed": seed, "generator": "fallback-stub"},
        ),
        SceneObject3D(
            id="table-1",
            category="table",
            position=(0.0, 0.0, 0.0),
            size=(1.0, 0.8, 1.0),
            orientation=0,
            metadata={"seed": seed, "generator": "fallback-stub"},
        ),
        SceneObject3D(
            id="chair-1",
            category="chair",
            position=(random.uniform(-1, 1), 0.0, random.uniform(-1, 1)),
            size=(0.6, 0.9, 0.6),
            orientation=random.choice([0, 1, 2, 3]),
            metadata={"seed": seed, "generator": "fallback-stub"},
        ),
    ]
    return base


@app.post("/generate-layout", response_model=LayoutResponse)
async def generate_layout(request: LayoutRequest):
    """
    Generate a semantic room layout from the given request and return a LayoutResponse.
    
    Attempts to fetch an architectural mask (if request.arch_mask_url is provided), encodes the request's vibe specification to derive a vibe bias, and invokes the semantic layout generator. If any step fails, returns a deterministic fallback response containing a dummy semantic PNG URL, a seeded list of scene objects, world_scale 0.01, and a default rectangular room outline.
    
    Parameters:
        request (LayoutRequest): Request containing room_type, optional arch_mask_url and mask_type, vibe_spec, and optional seed.
    
    Returns:
        LayoutResponse: The generated layout response. On success this contains the semantic_map_png_url (when available), generated SceneObject3D objects, world_scale, and optional room_outline; on failure these fields are populated with the deterministic fallback values described above.
    """
    seed = request.seed or 1

    arch_mask = None
    if request.arch_mask_url:
        try:
            resp = requests.get(request.arch_mask_url, timeout=5)
            resp.raise_for_status()
            from PIL import Image
            import io
            mask_img = Image.open(io.BytesIO(resp.content)).convert("L")
            arch_mask = np.array(mask_img)
        except Exception:
            arch_mask = None

    try:
        # Encode vibe
        vibe_encoding = vibe_encoder.encode_vibe_spec(request.vibe_spec.dict())
        category_bias = vibe_encoding.get("category_bias")

        semantic, instances = generate_semantic_layout(
            room_type=request.room_type,
            arch_mask=arch_mask,
            seed=seed,
            vibe_bias=category_bias,
        )
        resp = to_layout_response(semantic, instances)
        return LayoutResponse(**resp)
    except Exception:
        # Fallback stub on any failure
        objects = stub_objects(seed)
        dummy_semantic = np.zeros((256, 256), dtype=np.uint8)
        return LayoutResponse(
            semantic_map_png_url=semantic_to_png_url(dummy_semantic),
            objects=objects,
            world_scale=0.01,
            room_outline=[(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
