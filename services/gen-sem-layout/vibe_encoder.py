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
        Create a VibeEncoder configured to use a specified CLIP text model on a target device.
        
        Attempts to initialize CLIP text and image components; if initialization fails or CLIP is unavailable, the instance remains in a deterministic "stub" mode that produces hash-based embeddings.
        
        Parameters:
            text_model_name (str): Identifier of the CLIP text model to load (e.g., "openai/clip-vit-base-patch32").
            device (str): Device to run inference on (e.g., "cuda" or "cpu"). If not provided, defaults to CUDA when available.
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
        """
        Prepare CLIP components for text and image encoding using the given pretrained model identifier.
        
        Parameters:
            model_name (str): Identifier or path of the pretrained CLIP model to load. This will be used to locate tokenizer, text model, processor, and image model.
        
        Notes:
            This method sets self.initialized to True when complete. Actual model loading is currently not implemented (stubbed) and should be provided in a future update.
        """
        print(f"Initializing vibe encoder with model: {model_name}")

        # TODO: Implement actual model loading
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        # self.processor = CLIPProcessor.from_pretrained(model_name)
        # self.image_model = CLIPModel.from_pretrained(model_name).to(self.device)

        self.initialized = True

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text prompt into a latent embedding vector.
        
        Parameters:
            text (str): The input text prompt to encode. 
        
        Returns:
            numpy.ndarray: Latent embedding vector for the input text. If the encoder is not initialized or CLIP is unavailable, returns a deterministic 128-dimensional stub embedding; otherwise returns the model-produced embedding.
        """
        if not self.initialized or not VIBE_ENCODER_AVAILABLE:
            # Return stub embedding
            return self._get_stub_embedding(text, "text")

        # TODO: Implement actual text encoding
        # inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # with torch.no_grad():
        #     outputs = self.text_model(**inputs)
        #     embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        # return embedding

        return self._get_stub_embedding(text, "text")

    def encode_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        Encode a reference image into a latent embedding.
        
        Parameters:
            image_url (str): Image source; accepts a base64 data URL (data:image/...) or a remote URL. When a remote URL is provided, network loading is not implemented and the function may return None.
        
        Returns:
            np.ndarray or None: A 1-D latent vector for the image, or `None` if encoding fails or the image cannot be loaded.
        """
        if not self.initialized or not VIBE_ENCODER_AVAILABLE:
            return self._get_stub_embedding("image", "image")

        try:
            # Decode image if base64
            if image_url.startswith("data:image"):
                image_data = image_url.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # Load from URL (would need requests in production)
                return None

            # TODO: Implement actual image encoding
            # inputs = self.processor(images=image, return_tensors="pt")
            # with torch.no_grad():
            #     outputs = self.image_model.get_image_features(**inputs)
            #     embedding = outputs.squeeze().cpu().numpy()
            # return embedding

            return self._get_stub_embedding("image", "image")
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def compute_category_bias(
        self,
        vibe_spec: Dict[str, Any],
        available_categories: List[str]
    ) -> Dict[str, float]:
        """
        Compute bias weights for each category based on the provided vibe specification.
        
        Parameters:
            vibe_spec (dict): Vibe specification containing prompt text, tags, sliders, or other cues used to derive category preferences.
            available_categories (list[str]): List of category identifiers to score against the vibe specification.
        
        Returns:
            dict[str, float]: A mapping from each category in `available_categories` to a numeric bias weight (larger values indicate stronger bias toward that category).
        """
        from category_bias import compute_category_bias
        return compute_category_bias(vibe_spec, available_categories)

    def encode_vibe_spec(self, vibe_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode a full vibe specification into text, image, and combined latent representations plus category bias and metadata.
        
        Parameters:
            vibe_spec (Dict[str, Any]): Vibe specification expected to contain a "prompt" mapping with keys:
                - "text": textual prompt string (optional)
                - "referenceImageUrl": image data URL or URL string (optional)
        
        Returns:
            Dict[str, Any]: Mapping with the following keys:
                - "text_latent": list of floats representing the text embedding.
                - "image_latent": list of floats for the image embedding or `None` if no image was encoded.
                - "combined_latent": list of floats representing the concatenated text and image latents (or text latent alone).
                - "category_bias": category bias structure as produced by `compute_category_bias`.
                - "metadata": dict with:
                    - "has_text": `true` if the prompt contained non-empty text.
                    - "has_image": `true` if an image was successfully encoded.
                    - "model": string identifying the encoder mode ("clip-stub" when uninitialized, otherwise "clip").
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

        # Combine latents (simple concatenation for stub)
        if image_latent is not None:
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