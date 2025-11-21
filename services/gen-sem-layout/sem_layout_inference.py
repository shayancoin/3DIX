import base64
import io
import random
import uuid
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


Instance = Tuple[int, Tuple[float, float, float], Tuple[float, float, float], int]

# Simple semantic palette (category id -> RGB)
PALETTE = {
    0: (255, 255, 255),  # background
    1: (200, 200, 200),  # floor
    2: (160, 160, 160),  # wall
    3: (150, 111, 51),   # door
    4: (100, 149, 237),  # window
    5: (220, 20, 60),    # sofa
    6: (255, 215, 0),    # table
    7: (60, 179, 113),   # chair
}

CATEGORY_NAMES = {
    1: "floor",
    2: "wall",
    3: "door",
    4: "window",
    5: "sofa",
    6: "table",
    7: "chair",
}

ASSET_INDEX = [
    {"category": "sofa", "size": (2.0, 1.0, 1.0), "asset_url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb"},
    {"category": "table", "size": (1.0, 0.8, 1.0), "asset_url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/CesiumMan/glTF/CesiumMan.gltf"},
    {"category": "chair", "size": (0.6, 0.9, 0.6), "asset_url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Avocado/glTF-Binary/Avocado.glb"},
]


def _draw_semantic_map(instances: List[Instance], size: int = 256) -> np.ndarray:
    semantic = np.zeros((size, size), dtype=np.uint8)
    for inst in instances:
        cid, size3d, pos3d, _ = inst
        cx = int(size / 2 + pos3d[0] * 10)
        cz = int(size / 2 + pos3d[2] * 10)
        w = int(size3d[0] * 10)
        d = int(size3d[2] * 10)
        x0 = max(0, cx - w // 2)
        x1 = min(size, cx + w // 2)
        z0 = max(0, cz - d // 2)
        z1 = min(size, cz + d // 2)
        semantic[z0:z1, x0:x1] = cid
    return semantic


def semantic_to_png_url(semantic: np.ndarray) -> str:
    h, w = semantic.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in PALETTE.items():
        rgb[semantic == cid] = color
    img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_semantic_layout(
    room_type: str,
    arch_mask: Optional[np.ndarray],
    seed: int,
    vibe_bias: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[Instance]]:
    random.seed(seed)
    np.random.seed(seed)

    # stub instances; in real impl, call SemLayoutDiff and attribute network
    instances: List[Instance] = [
        (5, (2.0, 1.0, 1.0), (1.0, 0.0, 2.0), 1),  # sofa
        (6, (1.0, 0.8, 1.0), (0.0, 0.0, 0.0), 0),  # table
        (7, (0.6, 0.9, 0.6), (random.uniform(-1, 1), 0.0, random.uniform(-1, 1)), random.choice([0, 1, 2, 3])),
    ]

    semantic = _draw_semantic_map(instances)

    # Apply architecture mask if provided (simple overlay)
    if arch_mask is not None:
        arch_mask_resized = arch_mask
        if arch_mask.shape != semantic.shape:
            arch_mask_resized = np.array(Image.fromarray(arch_mask).resize(semantic.shape[::-1], resample=Image.NEAREST))
        semantic = np.maximum(semantic, arch_mask_resized.astype(np.uint8))

    return semantic, instances


def to_layout_response(semantic: np.ndarray, instances: List[Instance]):
    objects = []
    for cid, size3d, pos3d, orient in instances:
        # simple nearest-size retrieval
        category_name = CATEGORY_NAMES.get(cid, f"cat-{cid}")
        best_asset = None
        best_score = 1e9
        for asset in ASSET_INDEX:
            if asset["category"] != category_name:
                continue
            a_size = asset["size"]
            score = sum((size3d[i] - a_size[i]) ** 2 for i in range(3)) / 3
            if score < best_score:
                best_score = score
                best_asset = asset

        objects.append(
            {
                "id": str(uuid.uuid4()),
                "category": category_name,
                "position": [pos3d[0], pos3d[1], pos3d[2]],
                "size": [size3d[0], size3d[1], size3d[2]],
                "orientation": int(orient),
                "mesh_url": best_asset["asset_url"] if best_asset else None,
            }
        )

    room_outline = [
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 4.0),
        (0.0, 4.0),
    ]

    return {
        "semantic_map_png_url": semantic_to_png_url(semantic),
        "world_scale": 0.01,
        "objects": objects,
        "room_outline": room_outline,
        "metadata": {
            "generator": "sem-layout-stub",
        },
    }
