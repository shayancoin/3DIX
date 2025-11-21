"""
Asset Retrieval Service for 3D-FUTURE models.
This module handles retrieval of 3D furniture models based on category and size constraints.
"""

import os
import json
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add research code to path
RESEARCH_PATH = os.path.join(os.path.dirname(__file__), "../../research/sem-layout-diff")
if RESEARCH_PATH not in sys.path:
    sys.path.insert(0, RESEARCH_PATH)

try:
    from preprocess.threed_front.datasets.threed_future_dataset import ThreedFutureDataset
    ASSET_RETRIEVAL_AVAILABLE = True
except ImportError:
    ASSET_RETRIEVAL_AVAILABLE = False
    print("Warning: 3D-FUTURE dataset not available, using stub asset retrieval")


class AssetRetrieval:
    """Service for retrieving 3D furniture assets from 3D-FUTURE dataset."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        model_info_path: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize asset retrieval service.
        
        Args:
            dataset_path: Path to 3D-FUTURE dataset JSON file
            model_info_path: Path to 3D-FUTURE model_info.json
            base_url: Base URL for serving asset files (for web access)
        """
        self.dataset_path = dataset_path or os.getenv("THREED_FUTURE_DATASET_PATH")
        self.model_info_path = model_info_path or os.getenv("THREED_FUTURE_MODEL_INFO_PATH")
        self.base_url = base_url or os.getenv("ASSET_BASE_URL", "http://localhost:8001/assets")
        self.dataset = None
        self.asset_index: Dict[str, List[Dict]] = {}

        if ASSET_RETRIEVAL_AVAILABLE and self.dataset_path:
            try:
                self._load_dataset()
            except Exception as e:
                print(f"Warning: Failed to load 3D-FUTURE dataset: {e}")
                print("Falling back to stub asset retrieval")

    def _load_dataset(self):
        """Load 3D-FUTURE dataset."""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return

        try:
            self.dataset = ThreedFutureDataset.from_dataset_file(self.dataset_path)
            self._build_index()
            print(f"Loaded 3D-FUTURE dataset with {len(self.dataset)} objects")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def _build_index(self):
        """Build index of assets by category."""
        if not self.dataset:
            return

        self.asset_index = {}
        for obj in self.dataset:
            category = obj.label
            if category not in self.asset_index:
                self.asset_index[category] = []

            # Store asset metadata
            asset_info = {
                "model_id": obj.model_id,
                "category": category,
                "size": obj.size.tolist() if hasattr(obj.size, 'tolist') else list(obj.size),
                "model_path": getattr(obj, 'raw_model_path', None),
                "texture_path": getattr(obj, 'texture_image_path', None),
            }
            self.asset_index[category].append(asset_info)

    def retrieve_asset(
        self,
        category: str,
        target_size: List[float],
        quality: str = "high"
    ) -> Optional[Dict]:
        """
        Retrieve an asset matching category and size constraints.
        
        Args:
            category: Object category (e.g., 'refrigerator', 'sink')
            target_size: Target size [width, height, depth]
            quality: Quality level ('low', 'medium', 'high')
            
        Returns:
            Asset metadata dict with model information
        """
        if not self.dataset or category not in self.asset_index:
            # Return stub asset
            return self._get_stub_asset(category, target_size, quality)

        # Find best matching asset by size
        candidates = self.asset_index[category]
        if not candidates:
            return self._get_stub_asset(category, target_size, quality)

        # Calculate size similarity
        best_match = None
        best_score = float('inf')

        for asset in candidates:
            asset_size = asset["size"]
            # Calculate normalized size difference (L2 norm)
            # This gives better matching for objects of different scales
            size_diff = np.array([
                abs(asset_size[0] - target_size[0]) / max(target_size[0], 0.1),
                abs(asset_size[1] - target_size[1]) / max(target_size[1], 0.1),
                abs(asset_size[2] - target_size[2]) / max(target_size[2], 0.1),
            ])
            # Use L2 norm for better matching
            score = np.sqrt(np.sum(size_diff ** 2))

            if score < best_score:
                best_score = score
                best_match = asset

        if best_match:
            # Generate asset URL
            model_id = best_match["model_id"]
            asset_url = f"{self.base_url}/{model_id}/model.gltf"
            if quality == "low":
                asset_url = f"{self.base_url}/{model_id}/model_low.gltf"
            elif quality == "medium":
                asset_url = f"{self.base_url}/{model_id}/model_medium.gltf"

            return {
                "modelId": model_id,
                "category": category,
                "url": asset_url,
                "size": best_match["size"],
                "textureUrl": f"{self.base_url}/{model_id}/texture.jpg" if best_match.get("texture_path") else None,
                "quality": quality,
            }

        return self._get_stub_asset(category, target_size, quality)

    def _get_stub_asset(
        self,
        category: str,
        target_size: List[float],
        quality: str
    ) -> Dict:
        """Generate stub asset information."""
        return {
            "modelId": f"stub-{category}-{hash(tuple(target_size)) % 10000}",
            "category": category,
            "url": f"{self.base_url}/stub/{category}.gltf",
            "size": target_size,
            "textureUrl": None,
            "quality": quality,
            "isStub": True,
        }

    def get_available_categories(self) -> List[str]:
        """Get list of available asset categories."""
        if self.asset_index:
            return list(self.asset_index.keys())
        return ["refrigerator", "sink", "stove", "cabinet", "bed", "dresser", "table", "chair"]

    def retrieve_assets_for_layout(
        self,
        layout_objects: List[Dict],
        quality: str = "high"
    ) -> List[Dict]:
        """
        Retrieve assets for all objects in a layout.
        
        Args:
            layout_objects: List of layout objects with category and size
            quality: Quality level for all assets
            
        Returns:
            List of asset metadata for each object
        """
        assets = []
        for obj in layout_objects:
            category = obj.get("category", "furniture")
            size = obj.get("size", [1.0, 1.0, 1.0])
            
            # Ensure size is a list of 3 floats
            if isinstance(size, (list, tuple)) and len(size) >= 3:
                size = [float(size[0]), float(size[1]), float(size[2])]
            else:
                size = [1.0, 1.0, 1.0]
            
            asset = self.retrieve_asset(category, size, quality)
            if asset:
                # Store object ID for later matching
                obj_id = obj.get("id")
                if obj_id:
                    asset["objectId"] = obj_id
                assets.append(asset)
        return assets
