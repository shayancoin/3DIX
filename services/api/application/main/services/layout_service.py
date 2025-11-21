"""
Service for interacting with the ML layout generation service.
"""

import httpx
import os
from typing import Dict, Any, Optional


class LayoutService:
    """Client for the ML layout generation service."""

    def __init__(self, ml_service_url: Optional[str] = None):
        self.ml_service_url = ml_service_url or os.getenv(
            "ML_SERVICE_URL",
            "http://localhost:8001"
        )

    async def generate_layout(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request a layout from the ML service using the provided request payload.
        
        Sends `request_data` as JSON to the service's /generate-layout endpoint and returns the service's JSON response parsed to a dictionary.
        
        Parameters:
            request_data (Dict[str, Any]): Payload describing the layout request to send to the ML service.
        
        Returns:
            Dict[str, Any]: Parsed JSON response from the ML layout service.
        
        Raises:
            Exception: If the HTTP request fails; the exception message is prefixed with "ML service error:" and wraps the underlying httpx.HTTPError.
        """
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    f"{self.ml_service_url}/generate-layout",
                    json=request_data,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"ML service error: {str(e)}")

    async def health_check(self) -> bool:
        """Check if ML service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ml_service_url}/health")
                return response.status_code == 200
        except Exception:
            return False