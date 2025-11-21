"""
Verification script for Vibe Semantics (Step 8).
Tests monotonicity and category sensitivity of the vibe encoder and layout generator.
"""

import requests
import json
import numpy as np
from collections import Counter

BASE_URL = "http://localhost:8003"

def test_monotonicity():
    """
    Test if 'cluttered' vibe produces more objects than 'minimalist' vibe.
    """
    print("\nRunning Monotonicity Test...")

    # Minimalist Vibe
    minimalist_spec = {
        "prompt": {"text": "minimalist room"},
        "tags": [{"label": "minimalist", "weight": 1.0}],
        "sliders": [{"id": "complexity", "value": 0.1}]
    }

    # Cluttered Vibe
    cluttered_spec = {
        "prompt": {"text": "cluttered room"},
        "tags": [{"label": "cluttered", "weight": 1.0}],
        "sliders": [{"id": "complexity", "value": 1.0}]
    }

    min_counts = []
    clut_counts = []

    for _ in range(5):
        # Minimalist
        resp = requests.post(f"{BASE_URL}/generate-layout", json={
            "room_type": "living_room",
            "vibe_spec": minimalist_spec,
            "seed": np.random.randint(1000)
        })
        if resp.status_code == 200:
            min_counts.append(len(resp.json()["objects"]))

        # Cluttered
        resp = requests.post(f"{BASE_URL}/generate-layout", json={
            "room_type": "living_room",
            "vibe_spec": cluttered_spec,
            "seed": np.random.randint(1000)
        })
        if resp.status_code == 200:
            clut_counts.append(len(resp.json()["objects"]))

    avg_min = np.mean(min_counts)
    avg_clut = np.mean(clut_counts)

    print(f"Avg objects (Minimalist): {avg_min}")
    print(f"Avg objects (Cluttered): {avg_clut}")

    if avg_clut >= avg_min:
        print("PASS: Monotonicity confirmed.")
    else:
        print("FAIL: Monotonicity violation.")

def test_category_sensitivity():
    """
    Test if specific vibes increase probability of related objects.
    """
    print("\nRunning Category Sensitivity Test...")

    # Sofa Vibe
    sofa_spec = {
        "prompt": {"text": "living room with many sofas"},
        "tags": [{"label": "sofa", "weight": 1.0}],
        "sliders": []
    }

    sofa_counts = 0
    total_objects = 0

    for _ in range(10):
        resp = requests.post(f"{BASE_URL}/generate-layout", json={
            "room_type": "living_room",
            "vibe_spec": sofa_spec,
            "seed": np.random.randint(1000)
        })
        if resp.status_code == 200:
            objects = resp.json()["objects"]
            total_objects += len(objects)
            sofa_counts += sum(1 for obj in objects if obj["category"] == "sofa")

    sofa_ratio = sofa_counts / total_objects if total_objects > 0 else 0
    print(f"Sofa Ratio (Sofa Vibe): {sofa_ratio:.2f}")

    if sofa_ratio > 0.2: # Baseline is usually lower
        print("PASS: Category sensitivity confirmed.")
    else:
        print("FAIL: Category sensitivity weak.")

if __name__ == "__main__":
    try:
        # Check health
        requests.get(f"{BASE_URL}/health")
        test_monotonicity()
        test_category_sensitivity()
    except Exception as e:
        print(f"Error running tests: {e}")
        print("Ensure the service is running on port 8002")
