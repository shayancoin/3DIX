import requests
import json
import sys
import time

API_URL = "http://localhost:8005/api/v1"
SAM3D_URL = "http://localhost:8004"

def test_sam3d_service_direct():
    """
    Verify that the SAM-3D reconstruction service returns mesh and preview URLs for a sample image.
    
    Sends a reconstruction request for a sample image with a category hint and target size, asserts that the response JSON contains the keys "mesh_url" and "preview_png_url", and exits the process with status 1 if the check fails.
    """
    print("Testing SAM-3D Service Direct...")
    try:
        payload = {
            "image_url": "http://example.com/chair.jpg",
            "category_hint": "chair",
            "target_size": [0.5, 1.0, 0.5]
        }
        response = requests.post(f"{SAM3D_URL}/reconstruct", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"SUCCESS: SAM-3D Service returned: {data}")
        assert "mesh_url" in data
        assert "preview_png_url" in data
    except Exception as e:
        print(f"FAILURE: SAM-3D Service test failed: {e}")
        sys.exit(1)

def test_api_integration():
    """
    Test API integration for generating a custom mesh from an image.
    
    Sends a request to the platform API's custom-mesh endpoint for a test room/object and asserts the response includes `mesh_url` and `preview_png_url`; also verifies the returned `mesh_url` contains "Duck.glb". Exits the process with a non-zero status if the test fails.
    """
    print("\nTesting API Integration...")
    room_id = "test_room"
    object_id = "test_obj"

    try:
        payload = {
            "image_url": "http://example.com/my-custom-chair.jpg"
        }
        url = f"{API_URL}/rooms/{room_id}/objects/{object_id}/custom-mesh"
        print(f"POST {url}")
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            print(f"Error Response: {response.text}")

        response.raise_for_status()
        data = response.json()
        print(f"SUCCESS: API returned: {data}")
        assert "mesh_url" in data
        assert "preview_png_url" in data
        # Verify it returns the stubbed duck model
        assert "Duck.glb" in data["mesh_url"]
    except Exception as e:
        print(f"FAILURE: API Integration test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Verification for Step 9...")
    test_sam3d_service_direct()
    test_api_integration()
    print("\nALL TESTS PASSED!")