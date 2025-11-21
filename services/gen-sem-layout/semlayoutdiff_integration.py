"""
Integration layer for SemLayoutDiff model inference.
This module provides a clean interface to the SemLayoutDiff research code.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import io
import base64

# Add research code to path
RESEARCH_PATH = os.path.join(os.path.dirname(__file__), "../../research/sem-layout-diff")
if RESEARCH_PATH not in sys.path:
    sys.path.insert(0, RESEARCH_PATH)

try:
    from semlayoutdiff.sldn.sampling_utils import LayoutSampler
    from semlayoutdiff.apm.attr_module import FurnitureAttributesModel
    from semlayoutdiff.apm.utils import get_instance_masks, convert_2d_to_3d
    from semlayoutdiff.apm.inference_utils import load_json, process_semantic_map
    SEMLAYOUTDIFF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SemLayoutDiff not available: {e}")
    SEMLAYOUTDIFF_AVAILABLE = False


class SemLayoutDiffIntegration:
    """Integration wrapper for SemLayoutDiff models."""

    def __init__(
        self,
        sldn_checkpoint_path: Optional[str] = None,
        apm_checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SemLayoutDiff integration.
        
        Args:
            sldn_checkpoint_path: Path to SLDN checkpoint
            apm_checkpoint_path: Path to APM checkpoint
            config_path: Path to configuration files
            device: Device to run inference on
        """
        self.device = device
        self.sldn_model = None
        self.apm_model = None
        self.sldn_sampler = None
        self.config = None
        self.initialized = False

        if not SEMLAYOUTDIFF_AVAILABLE:
            print("SemLayoutDiff not available, using stub mode")
            return

        # Try to initialize models if checkpoints are provided
        if sldn_checkpoint_path and apm_checkpoint_path:
            try:
                self._initialize_models(sldn_checkpoint_path, apm_checkpoint_path, config_path)
            except Exception as e:
                print(f"Warning: Failed to initialize SemLayoutDiff models: {e}")
                print("Falling back to stub mode")

    def _initialize_models(
        self,
        sldn_checkpoint_path: str,
        apm_checkpoint_path: str,
        config_path: Optional[str] = None
    ):
        """Initialize SLDN and APM models."""
        # This would load the actual models
        # For now, we'll create a structure that can be connected
        print(f"Initializing SemLayoutDiff models...")
        print(f"SLDN checkpoint: {sldn_checkpoint_path}")
        print(f"APM checkpoint: {apm_checkpoint_path}")
        
        # TODO: Implement actual model loading
        # self.sldn_sampler = LayoutSampler(...)
        # self.apm_model = FurnitureAttributesModel.load_from_checkpoint(...)
        
        self.initialized = True

    def generate_semantic_layout(
        self,
        room_type: str,
        floor_plan_mask: Optional[np.ndarray] = None,
        num_samples: int = 1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate semantic layout using SLDN.
        
        Args:
            room_type: Room type (kitchen, bedroom, etc.)
            floor_plan_mask: Optional floor plan mask image
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (semantic_map, metadata)
        """
        if not self.initialized or not SEMLAYOUTDIFF_AVAILABLE:
            # Return stub semantic map
            return self._generate_stub_semantic_map(room_type), {
                "model": "stub",
                "room_type": room_type
            }

        # TODO: Implement actual SLDN inference
        # semantic_map = self.sldn_sampler.sample(...)
        # return semantic_map, metadata
        
        return self._generate_stub_semantic_map(room_type), {
            "model": "stub",
            "room_type": room_type
        }

    def predict_attributes(
        self,
        semantic_map: np.ndarray,
        room_type: str
    ) -> List[Dict[str, Any]]:
        """
        Predict 3D object attributes from semantic map using APM.
        
        Args:
            semantic_map: 2D semantic layout map
            room_type: Room type for category mapping
            
        Returns:
            List of object predictions with attributes
        """
        if not self.initialized or not SEMLAYOUTDIFF_AVAILABLE:
            return self._generate_stub_attributes(room_type, semantic_map)

        # TODO: Implement actual APM inference
        # Load required configs
        # instances_data = get_instance_masks(...)
        # predictions = self.apm_model.predict(...)
        # return formatted_predictions
        
        return self._generate_stub_attributes(room_type, semantic_map)

    def _generate_stub_semantic_map(self, room_type: str) -> np.ndarray:
        """Generate a stub semantic map for testing."""
        # Create a simple semantic map (256x256)
        size = 256
        semantic_map = np.zeros((size, size), dtype=np.uint8)
        
        # Add some basic structure based on room type
        if room_type == "kitchen":
            # Add some furniture regions
            semantic_map[100:150, 50:100] = 1  # Refrigerator
            semantic_map[100:150, 150:200] = 2  # Sink
            semantic_map[100:150, 200:250] = 3  # Stove
        elif room_type == "bedroom":
            semantic_map[80:180, 80:180] = 4  # Bed
            semantic_map[50:100, 200:250] = 5  # Dresser
        
        return semantic_map

    def _generate_stub_attributes(
        self,
        room_type: str,
        semantic_map: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate stub attributes from semantic map."""
        # Extract basic information from semantic map
        height, width = semantic_map.shape
        
        # Find unique categories
        unique_categories = np.unique(semantic_map)
        unique_categories = unique_categories[unique_categories > 0]  # Exclude background
        
        predictions = []
        category_map = {
            1: "refrigerator",
            2: "sink",
            3: "stove",
            4: "bed",
            5: "dresser",
        }
        
        for cat_id in unique_categories:
            # Find bounding box for this category
            mask = (semantic_map == cat_id)
            coords = np.where(mask)
            
            if len(coords[0]) == 0:
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Convert to 3D coordinates (normalize to room dimensions)
            center_x = (x_min + x_max) / 2.0 / width * 5.0  # Assume 5m width
            center_z = (y_min + y_max) / 2.0 / height * 4.0  # Assume 4m length
            center_y = 0.0
            
            # Estimate size
            size_x = (x_max - x_min) / width * 5.0
            size_z = (y_max - y_min) / height * 4.0
            size_y = 0.5  # Default height
            
            category = category_map.get(int(cat_id), "furniture")
            
            predictions.append({
                "category": category,
                "position": [center_x, center_y, center_z],
                "size": [size_x, size_y, size_z],
                "orientation": 0.0,
            })
        
        return predictions if predictions else self._get_default_objects(room_type)

    def _get_default_objects(self, room_type: str) -> List[Dict[str, Any]]:
        """Get default objects for room type."""
        defaults = {
            "kitchen": [
                {"category": "refrigerator", "position": [0.5, 0.0, 0.3], "size": [0.6, 1.8, 0.6], "orientation": 0.0},
                {"category": "sink", "position": [2.0, 0.0, 2.0], "size": [0.6, 0.3, 0.6], "orientation": 1.57},
                {"category": "stove", "position": [3.5, 0.0, 2.0], "size": [0.6, 0.3, 0.6], "orientation": 1.57},
            ],
            "bedroom": [
                {"category": "bed", "position": [2.5, 0.0, 2.0], "size": [2.0, 0.5, 1.8], "orientation": 0.0},
                {"category": "dresser", "position": [0.5, 0.0, 0.5], "size": [1.2, 1.0, 0.5], "orientation": 1.57},
            ],
        }
        return defaults.get(room_type, defaults["kitchen"])

    def semantic_map_to_base64(self, semantic_map: np.ndarray) -> str:
        """Convert semantic map to base64 encoded image."""
        # Normalize to 0-255
        normalized = ((semantic_map - semantic_map.min()) / 
                      (semantic_map.max() - semantic_map.min() + 1e-8) * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(normalized, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
