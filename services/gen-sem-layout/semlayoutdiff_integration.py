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
import colorsys

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
        print(f"Initializing SemLayoutDiff models...")
        print(f"SLDN checkpoint: {sldn_checkpoint_path}")
        print(f"APM checkpoint: {apm_checkpoint_path}")
        
        try:
            # Initialize APM model (simpler, can be done first)
            if apm_checkpoint_path and os.path.exists(apm_checkpoint_path):
                # Load APM model using PyTorch Lightning
                # Note: This requires a config file - we'll use a default structure
                from omegaconf import DictConfig, OmegaConf
                
                # Create minimal config for APM if not provided
                if config_path and os.path.exists(config_path):
                    apm_config = OmegaConf.load(config_path)
                else:
                    # Use default config structure
                    apm_config = OmegaConf.create({
                        'model': {
                            'semantic_encoder': {},
                            'num_categories': 38,
                            'embedding_dim': 256
                        },
                        'trainer': {
                            'lr': 0.001,
                            'l1_lambda': 0.1,
                            'l2_lambda': 0.1
                        }
                    })
                
                self.apm_model = FurnitureAttributesModel.load_from_checkpoint(
                    apm_checkpoint_path,
                    config=apm_config,
                    strict=False
                )
                self.apm_model.eval()
                self.apm_model.freeze()
                if torch.cuda.is_available():
                    self.apm_model = self.apm_model.cuda()
                print("APM model loaded successfully")
            else:
                print(f"APM checkpoint not found at {apm_checkpoint_path}, using stub mode")
            
            # Initialize SLDN sampler (more complex, requires config)
            if sldn_checkpoint_path and os.path.exists(sldn_checkpoint_path):
                # SLDN requires a full config with model path, etc.
                # For now, we'll mark it as available but defer full initialization
                # until we have proper config files
                print("SLDN checkpoint found, but full initialization requires config files")
                print("Using stub mode for SLDN until proper config is provided")
                # self.sldn_sampler = LayoutSampler(cfg)  # Requires DictConfig
            else:
                print(f"SLDN checkpoint not found at {sldn_checkpoint_path}, using stub mode")
            
            # Mark as initialized if at least APM is loaded
            # Full initialization requires both models + configs
            if self.apm_model is not None:
                self.initialized = True
                print("SemLayoutDiff models initialized (partial - APM only)")
            else:
                print("SemLayoutDiff models not fully initialized, using stub mode")
                
        except Exception as e:
            print(f"Error initializing SemLayoutDiff models: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to stub mode")
            self.initialized = False

    def generate_semantic_layout(
        self,
        room_type: str,
        floor_plan_mask: Optional[np.ndarray] = None,
        num_samples: int = 1,
        text_embedding: Optional[np.ndarray] = None,
        category_bias: Optional[Dict[str, float]] = None
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
            # Return stub semantic map with category bias applied
            return self._generate_stub_semantic_map(room_type, category_bias), {
                "model": "stub",
                "room_type": room_type,
                "category_bias_applied": category_bias is not None
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
        room_type: str,
        category_bias: Optional[Dict[str, float]] = None
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

    def _generate_stub_semantic_map(self, room_type: str, category_bias: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Generate a stub semantic map for testing, with optional category bias."""
        # Create a semantic map matching SemLayoutDiff's expected size (1200x1200)
        # For stub mode, we'll use a smaller size (256x256) for performance
        size = 256
        semantic_map = np.zeros((size, size), dtype=np.uint8)
        
        # Add floor region (category 0 is typically background, 1 is floor)
        # Fill most of the map with floor
        semantic_map[20:size-20, 20:size-20] = 1  # Floor region
        
        # Category to ID mapping
        category_to_id = {
            "refrigerator": 2,
            "sink": 3,
            "stove": 4,
            "counter": 5,
            "bed": 6,
            "dresser": 7,
            "nightstand": 8,
            "toilet": 10,
            "shower": 11,
            "table": 12,
            "chair": 13,
        }
        
        # Apply category bias if provided
        if category_bias:
            # Sort categories by bias weight (highest first)
            sorted_categories = sorted(
                category_bias.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Generate objects based on bias weights
            object_count = max(3, min(8, int(sum(category_bias.values()) * 5)))
            positions = []
            
            for i, (category, bias) in enumerate(sorted_categories[:object_count]):
                if category in category_to_id and bias > 0.3:
                    cat_id = category_to_id[category]
                    # Place objects with spacing based on bias
                    y_pos = 60 + (i * 40) % (size - 120)
                    x_pos = 40 + (i * 60) % (size - 80)
                    
                    # Ensure no overlap
                    overlap = False
                    for py, px, pw, ph in positions:
                        if abs(y_pos - py) < 50 and abs(x_pos - px) < 50:
                            overlap = True
                            break
                    
                    if not overlap:
                        # Size based on bias (higher bias = larger object)
                        obj_size = int(30 + bias * 40)
                        semantic_map[y_pos:y_pos+obj_size, x_pos:x_pos+obj_size] = cat_id
                        positions.append((y_pos, x_pos, obj_size, obj_size))
        else:
            # Default layout based on room type
            if room_type == "kitchen":
                semantic_map[80:130, 30:80] = 2   # Refrigerator
                semantic_map[80:130, 100:150] = 3  # Sink
                semantic_map[80:130, 180:230] = 4  # Stove
                semantic_map[150:200, 50:200] = 5  # Counter
            elif room_type == "bedroom":
                semantic_map[60:180, 60:180] = 6   # Bed
                semantic_map[30:80, 200:250] = 7   # Dresser
                semantic_map[200:250, 30:80] = 8   # Nightstand
            elif room_type == "bathroom":
                semantic_map[80:130, 100:150] = 9   # Sink
                semantic_map[150:200, 50:100] = 10  # Toilet
                semantic_map[150:200, 150:200] = 11 # Shower
            else:
                semantic_map[80:150, 80:150] = 12   # Table
                semantic_map[180:230, 50:100] = 13  # Chair 1
                semantic_map[180:230, 150:200] = 14 # Chair 2
        
        return semantic_map

    def _generate_stub_attributes(
        self,
        room_type: str,
        semantic_map: np.ndarray,
        category_bias: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Generate stub attributes from semantic map."""
        # Extract basic information from semantic map
        height, width = semantic_map.shape
        
        # Find unique categories
        unique_categories = np.unique(semantic_map)
        unique_categories = unique_categories[unique_categories > 0]  # Exclude background
        
        predictions = []
        category_map = {
            1: "floor",  # Floor (background)
            2: "refrigerator",
            3: "sink",
            4: "stove",
            5: "counter",
            6: "bed",
            7: "dresser",
            8: "nightstand",
            9: "sink",  # Bathroom sink
            10: "toilet",
            11: "shower",
            12: "table",
            13: "chair",
            14: "chair",
        }
        
        for cat_id in unique_categories:
            # Skip floor/background (category 0 or 1)
            if int(cat_id) <= 1:
                continue
                
            # Find bounding box for this category
            mask = (semantic_map == cat_id)
            coords = np.where(mask)
            
            if len(coords[0]) == 0:
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Convert to 3D coordinates (normalize to room dimensions)
            # SemLayoutDiff uses 0.01 m/pixel scale, so 256 pixels = 2.56m
            # We'll scale to a standard 5m x 4m room
            # For stub mode, we assume the semantic map represents a 5m x 4m room
            room_width_m = 5.0
            room_length_m = 4.0
            
            # Convert pixel coordinates to world coordinates
            # Map from [0, width] to [-room_width_m/2, room_width_m/2]
            # Map from [0, height] to [-room_length_m/2, room_length_m/2]
            center_x = ((x_min + x_max) / 2.0 / width - 0.5) * room_width_m
            center_z = ((y_min + y_max) / 2.0 / height - 0.5) * room_length_m
            center_y = 0.0
            
            # Estimate size in meters (scale pixel dimensions to room dimensions)
            size_x = max((x_max - x_min) / width * room_width_m, 0.3)  # Minimum 0.3m
            size_z = max((y_max - y_min) / height * room_length_m, 0.3)  # Minimum 0.3m
            
            # Set appropriate height based on category
            category = category_map.get(int(cat_id), "furniture")
            height_map = {
                "refrigerator": 1.8,
                "sink": 0.8,
                "stove": 0.9,
                "counter": 0.9,
                "bed": 0.5,
                "dresser": 1.0,
                "nightstand": 0.6,
                "toilet": 0.4,
                "shower": 2.0,
                "table": 0.75,
                "chair": 0.9,
            }
            size_y = height_map.get(category, 0.8)
            
            predictions.append({
                "category": category,
                "position": [center_x, center_y, center_z],
                "size": [size_x, size_y, size_z],
                "orientation": 0.0,
            })
        
        # Apply category bias if provided
        if category_bias and predictions:
            # Sort predictions by bias weight (higher bias = more likely)
            predictions_with_bias = []
            for pred in predictions:
                category = pred["category"]
                bias = category_bias.get(category, 0.5)
                pred["bias_weight"] = bias
                predictions_with_bias.append((bias, pred))
            
            # Sort by bias (descending) and take top objects
            predictions_with_bias.sort(key=lambda x: x[0], reverse=True)
            # Take top 70% of objects based on bias, or all if bias is uniform
            num_to_keep = max(1, int(len(predictions_with_bias) * 0.7))
            predictions = [pred for _, pred in predictions_with_bias[:num_to_keep]]
        
        return predictions if predictions else self._get_default_objects(room_type, category_bias)

    def _get_default_objects(self, room_type: str, category_bias: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Get default objects for room type, optionally filtered by category bias."""
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
        
        objects = defaults.get(room_type, defaults["kitchen"])
        
        # Filter by category bias if provided
        if category_bias:
            filtered_objects = []
            for obj in objects:
                category = obj["category"]
                bias = category_bias.get(category, 0.5)
                # Include objects with bias > 0.3
                if bias > 0.3:
                    filtered_objects.append(obj)
            if filtered_objects:
                return filtered_objects
        
        return objects

    def semantic_map_to_base64(self, semantic_map: np.ndarray) -> str:
        """Convert semantic map to base64 encoded image with color palette."""
        # Create a color palette for visualization
        # Use distinct colors for different categories
        height, width = semantic_map.shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Define a color palette (RGB)
        # Category 0: black (background), Category 1: gray (floor)
        # Other categories: distinct colors
        palette = {
            0: (0, 0, 0),        # Black (background)
            1: (128, 128, 128),  # Gray (floor)
            2: (255, 0, 0),      # Red
            3: (0, 255, 0),      # Green
            4: (0, 0, 255),      # Blue
            5: (255, 255, 0),    # Yellow
            6: (255, 0, 255),    # Magenta
            7: (0, 255, 255),    # Cyan
            8: (255, 128, 0),    # Orange
            9: (128, 0, 255),    # Purple
            10: (255, 192, 203), # Pink
            11: (165, 42, 42),   # Brown
            12: (128, 128, 0),   # Olive
            13: (0, 128, 128),   # Teal
            14: (128, 0, 128),   # Maroon
        }
        
        # Apply color palette
        for cat_id, color in palette.items():
            mask = (semantic_map == cat_id)
            colored_map[mask] = color
        
        # For categories not in palette, use a hash-based color
        unique_cats = np.unique(semantic_map)
        for cat_id in unique_cats:
            if int(cat_id) not in palette:
                # Generate a deterministic color based on category ID
                hue = (int(cat_id) * 137) % 360  # Golden angle for color distribution
                rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.7, 0.9)
                color = tuple(int(c * 255) for c in rgb)
                mask = (semantic_map == cat_id)
                colored_map[mask] = color
        
        # Create PIL image
        img = Image.fromarray(colored_map, mode='RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
