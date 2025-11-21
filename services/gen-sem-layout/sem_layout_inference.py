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
    """
    Render a 2D semantic map from scene instances onto a square grid.

    Parameters:
        instances (List[Instance]): Sequence of instances, each as (category_id, size3d, position3d, orientation).
            Positions and sizes are interpreted in world units and projected onto the grid.
        size (int): Width and height of the output square semantic map in pixels.

    Returns:
        np.ndarray: A 2D uint8 array of shape (size, size) where each element is the semantic category id for that pixel.
    """
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
    """
    Convert a 2D semantic class map into a PNG data URL using the module PALETTE.

    Parameters:
        semantic (np.ndarray): 2D array of integer class IDs where each value selects a color from PALETTE.

    Returns:
        str: A data URL (`data:image/png;base64,...`) containing a PNG image in which each class ID is rendered with its corresponding RGB color from PALETTE.
    """
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
    vibe_bias: Optional[dict],
) -> Tuple[np.ndarray, List[Instance]]:
    """
    Generate a deterministic semantic occupancy map and a list of scene instances for a simple room layout.

    The function produces a 2D semantic map where each pixel encodes a category id and a corresponding list of instances describing placed objects. If an architecture mask is provided, it is overlaid onto the semantic map; the mask is resized to match the semantic shape using nearest-neighbor interpolation when necessary.

    Parameters:
        room_type (str): High-level room type hint (e.g., "living_room") used to influence layout priors.
        arch_mask (Optional[np.ndarray]): Optional 2D array of category ids to overlay on the generated semantic map.
            If its shape differs from the generated map, it is resized with nearest-neighbor interpolation before overlay.
        seed (int): Seed for randomness to ensure deterministic outputs.
        vibe_bias (Optional[dict]): Optional bias dictionary that can influence stochastic placement decisions.

    Returns:
        semantic (np.ndarray): 2D array (uint8) where each value is a semantic category id for that pixel.
        instances (List[Instance]): List of placed instances; each element is a tuple (category_id, size3d, pos3d, orientation).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Default bias if none provided
    if vibe_bias is None:
        vibe_bias = {cat: 0.5 for cat in ["sofa", "table", "chair"]}

    # Convert bias dict to simple lookup if needed, or assume it's a dict
    # In this stub, we expect vibe_bias to be a dict of category -> probability weight

    instances: List[Instance] = []

    # Probabilistic generation based on bias
    # Sofa
    if random.random() < vibe_bias.get("sofa", 0.5) * 1.5: # Boost base prob
        instances.append((5, (2.0, 1.0, 1.0), (1.0, 0.0, 2.0), 1))

    # Table
    if random.random() < vibe_bias.get("table", 0.5) * 1.5:
        instances.append((6, (1.0, 0.8, 1.0), (0.0, 0.0, 0.0), 0))

    # Chairs - number depends on bias
    chair_bias = vibe_bias.get("chair", 0.5)
    num_chairs = 0
    if chair_bias > 0.7:
        num_chairs = random.randint(2, 4)
    elif chair_bias > 0.3:
        num_chairs = random.randint(1, 2)

    for _ in range(num_chairs):
        instances.append((7, (0.6, 0.9, 0.6), (random.uniform(-1, 1), 0.0, random.uniform(-1, 1)), random.choice([0, 1, 2, 3])))

    # If nothing generated, add at least one chair
    if not instances:
        instances.append((7, (0.6, 0.9, 0.6), (0.0, 0.0, 0.0), 0))

    semantic = _draw_semantic_map(instances)

    # Apply architecture mask if provided (simple overlay)
    if arch_mask is not None:
        arch_mask_resized = arch_mask
        if arch_mask.shape != semantic.shape:
            arch_mask_resized = np.array(Image.fromarray(arch_mask).resize(semantic.shape[::-1], resample=Image.NEAREST))
        semantic = np.maximum(semantic, arch_mask_resized.astype(np.uint8))

    return semantic, instances


def to_layout_response(semantic: np.ndarray, instances: List[Instance]):
    """
    Builds a layout response dictionary containing a PNG data URL of the semantic map, world scale, scene objects, a room outline, and generator metadata.

    Parameters:
        semantic (np.ndarray): 2D semantic map array where integer values represent semantic class ids.
        instances (List[Instance]): List of instances as (category_id, size3d, pos3d, orientation).

    Returns:
        dict: A response object with the following keys:
            - semantic_map_png_url (str): PNG data URL generated from `semantic`.
            - world_scale (float): Scale factor applied to world coordinates.
            - objects (List[dict]): List of object entries, each containing:
                - id (str): UUID for the object.
                - category (str): Human-readable category name.
                - position (List[float]): [x, y, z] world position.
                - size (List[float]): [w, h, d] object size.
                - orientation (int): Object orientation as an integer.
                - mesh_url (str|None): URL to a matching asset mesh, or `None` if none found.
            - room_outline (List[Tuple[float, float]]): Polygon points describing the room boundary.
            - metadata (dict): Additional metadata; includes `"generator": "sem-layout-stub"`.
    """
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
