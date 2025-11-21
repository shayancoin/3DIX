"""
Vibe Encoder Service
Encodes text and image inputs to latent representations for layout generation.
Implements category bias learning to influence object category selection.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import io
import base64

# Add research code to path
RESEARCH_PATH = os.path.join(os.path.dirname(__file__), "../../research/sem-layout-diff")
if RESEARCH_PATH not in sys.path:
    sys.path.insert(0, RESEARCH_PATH)

try:
    # Try to import text embedding utilities
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
    VIBE_ENCODER_AVAILABLE = True
except ImportError:
    VIBE_ENCODER_AVAILABLE = False
    print("Warning: CLIP models not available, using stub vibe encoder")


class VibeEncoder:
    """Encodes vibe specifications to latent representations."""

    def __init__(
        self,
        text_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize vibe encoder.
        
        Args:
            text_model_name: Name of the text encoder model
            device: Device to run inference on
        """
        self.device = device
        self.text_model = None
        self.image_model = None
        self.tokenizer = None
        self.processor = None
        self.initialized = False

        if VIBE_ENCODER_AVAILABLE:
            try:
                self._initialize_models(text_model_name)
            except Exception as e:
                print(f"Warning: Failed to initialize vibe encoder models: {e}")
                print("Falling back to stub mode")

    def _initialize_models(self, model_name: str):
        """Initialize CLIP models for text and image encoding."""
        print(f"Initializing vibe encoder with model: {model_name}")
        
        try:
            if VIBE_ENCODER_AVAILABLE:
                # Load CLIP model for text and image encoding
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model.eval()
                print(f"CLIP model loaded successfully on {self.device}")
            else:
                print("CLIP models not available, using stub mode")
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            print("Falling back to stub mode")
        
        self.initialized = True

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text prompt to latent representation.
        
        Args:
            text: Text prompt
            
        Returns:
            Latent vector
        """
        if not text or not text.strip():
            return self._get_stub_embedding("", "text")
        
        if not self.initialized or not VIBE_ENCODER_AVAILABLE or not hasattr(self, 'model'):
            # Return stub embedding
            return self._get_stub_embedding(text, "text")

        try:
            # Encode text using CLIP
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embedding = text_features.squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error encoding text with CLIP: {e}, using stub")
            return self._get_stub_embedding(text, "text")

    def encode_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        Encode reference image to latent representation.
        
        Args:
            image_url: URL or base64 data URL of the image
            
        Returns:
            Latent vector or None if encoding fails
        """
        if not self.initialized or not VIBE_ENCODER_AVAILABLE or not hasattr(self, 'model'):
            return self._get_stub_embedding("image", "image")

        try:
            # Decode image if base64
            if image_url.startswith("data:image"):
                image_data = image_url.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                # Load from URL (would need requests in production)
                import requests
                try:
                    response = requests.get(image_url, timeout=10)
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    print(f"Error loading image from URL: {e}")
                    return None

            # Encode image using CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error encoding image with CLIP: {e}")
            return self._get_stub_embedding("image", "image")

    def compute_category_bias(
        self,
        vibe_spec: Dict[str, Any],
        available_categories: List[str]
    ) -> Dict[str, float]:
        """
        Compute category bias weights based on vibe specification.
        
        Args:
            vibe_spec: Vibe specification with tags, sliders, etc.
            available_categories: List of available object categories
            
        Returns:
            Dictionary mapping category names to bias weights (0-1)
        """
        bias_weights = {cat: 0.5 for cat in available_categories}  # Default neutral

        # Extract tags
        tags = vibe_spec.get("tags", [])
        prompt_text = vibe_spec.get("prompt", {}).get("text", "").lower()
        sliders = vibe_spec.get("sliders", [])

        # Category mappings based on tags and prompts
        category_keywords = {
            "refrigerator": ["refrigerator", "fridge", "kitchen", "appliance"],
            "sink": ["sink", "kitchen", "bathroom", "faucet"],
            "stove": ["stove", "oven", "cooktop", "kitchen"],
            "cabinet": ["cabinet", "storage", "kitchen"],
            "bed": ["bed", "bedroom", "sleep", "mattress"],
            "dresser": ["dresser", "bedroom", "storage", "drawer"],
            "table": ["table", "dining", "desk"],
            "chair": ["chair", "seating", "dining"],
            "sofa": ["sofa", "couch", "living", "seating"],
            "toilet": ["toilet", "bathroom"],
            "shower": ["shower", "bathroom"],
        }

        # Compute bias from tags
        for tag in tags:
            tag_label = tag.get("label", "").lower()
            tag_weight = tag.get("weight", 0.5)
            
            for category, keywords in category_keywords.items():
                if any(keyword in tag_label for keyword in keywords):
                    # Increase bias for matching categories
                    bias_weights[category] = min(1.0, bias_weights[category] + tag_weight * 0.2)

        # Compute bias from prompt text
        for category, keywords in category_keywords.items():
            if any(keyword in prompt_text for keyword in keywords):
                bias_weights[category] = min(1.0, bias_weights[category] + 0.3)

        # Adjust based on sliders
        for slider in sliders:
            slider_id = slider.get("id", "")
            slider_value = slider.get("value", 0.5)
            
            if slider_id == "complexity" or slider_id == "clutter":
                # Higher complexity/clutter = more objects, higher density
                for category in bias_weights:
                    # Increase bias proportionally to slider value
                    increase = (slider_value - 0.5) * 0.4  # Scale from -0.2 to +0.2
                    bias_weights[category] = np.clip(bias_weights[category] + increase, 0.0, 1.0)
            elif slider_id == "spaciousness" or slider_id == "openness":
                # Higher spaciousness/openness = fewer, larger objects
                if slider_value > 0.6:
                    # Reduce bias for small objects
                    small_objects = ["chair", "table", "nightstand"]
                    reduction = (slider_value - 0.6) * 0.5  # Scale from 0 to 0.2
                    for cat in small_objects:
                        if cat in bias_weights:
                            bias_weights[cat] = max(0.1, bias_weights[cat] - reduction)
            elif slider_id == "warmth":
                # Higher warmth = more soft furniture (sofas, beds, chairs)
                if slider_value > 0.6:
                    warm_categories = ["sofa", "bed", "chair"]
                    increase = (slider_value - 0.6) * 0.3
                    for cat in warm_categories:
                        if cat in bias_weights:
                            bias_weights[cat] = min(1.0, bias_weights[cat] + increase)

        # Normalize weights
        max_weight = max(bias_weights.values()) if bias_weights.values() else 1.0
        if max_weight > 0:
            bias_weights = {k: v / max_weight for k, v in bias_weights.items()}

        return bias_weights

    def encode_vibe_spec(self, vibe_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode complete vibe specification to latent representation.
        
        Args:
            vibe_spec: Complete vibe specification
            
        Returns:
            Dictionary with encoded latents and metadata
        """
        prompt = vibe_spec.get("prompt", {})
        text = prompt.get("text", "")
        image_url = prompt.get("referenceImageUrl")

        # Encode text
        text_latent = self.encode_text(text)

        # Encode image if provided
        image_latent = None
        if image_url:
            image_latent = self.encode_image(image_url)

        # Combine latents (weighted average for CLIP, concatenation for stub)
        if image_latent is not None:
            if len(text_latent) == len(image_latent):
                # Weighted average for same-dimensional embeddings (CLIP)
                combined_latent = 0.7 * text_latent + 0.3 * image_latent
            else:
                # Concatenation for different dimensions (stub)
                combined_latent = np.concatenate([text_latent, image_latent])
        else:
            combined_latent = text_latent

        # Compute category bias
        available_categories = [
            "refrigerator", "sink", "stove", "cabinet",
            "bed", "dresser", "table", "chair", "sofa",
            "toilet", "shower"
        ]
        category_bias = self.compute_category_bias(vibe_spec, available_categories)

        return {
            "text_latent": text_latent.tolist(),
            "image_latent": image_latent.tolist() if image_latent is not None else None,
            "combined_latent": combined_latent.tolist(),
            "category_bias": category_bias,
            "metadata": {
                "has_text": bool(text),
                "has_image": image_latent is not None,
                "model": "clip-stub" if not self.initialized else "clip",
            }
        }

    def _get_stub_embedding(self, content: str, type: str) -> np.ndarray:
        """Generate stub embedding."""
        # Create a simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(f"{type}:{content}".encode())
        hash_bytes = hash_obj.digest()
        # Convert to numpy array (128-dim for stub)
        embedding = np.frombuffer(hash_bytes * 4, dtype=np.float32)[:128]
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
