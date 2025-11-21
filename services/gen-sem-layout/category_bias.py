"""
Category Bias Logic
Computes bias weights for object categories based on vibe specifications.
"""

from typing import Dict, List, Any

def compute_category_bias(
    vibe_spec: Dict[str, Any],
    available_categories: List[str]
) -> Dict[str, float]:
    """
    Compute normalized bias weights for object categories from a vibe specification.
    
    Parameters:
        vibe_spec (dict): Vibe specification that may include:
            - "tags": list of tag objects with "label" (string) and optional "weight" (float).
            - "prompt": object with "text" (string).
            - "sliders": list of slider objects with "id" (string) and "value" (float).
        available_categories (list[str]): Categories to compute biases for.
    
    Returns:
        dict[str, float]: Mapping from each category name to a bias weight in [0.0, 1.0].
            Weights are normalized so that the maximum value is 1.0.
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

        if slider_id == "complexity":
            # Higher complexity = more objects
            for category in bias_weights:
                if slider_value > 0.7:
                    bias_weights[category] = min(1.0, bias_weights[category] + 0.1)
        elif slider_id == "spaciousness":
            # Higher spaciousness = fewer, larger objects
            if slider_value > 0.7:
                # Reduce bias for small objects
                small_objects = ["chair", "table", "nightstand"]
                for cat in small_objects:
                    if cat in bias_weights:
                        bias_weights[cat] = max(0.2, bias_weights[cat] - 0.2)

    # Normalize weights
    max_weight = max(bias_weights.values()) if bias_weights.values() else 1.0
    if max_weight > 0:
        bias_weights = {k: v / max_weight for k, v in bias_weights.items()}

    return bias_weights