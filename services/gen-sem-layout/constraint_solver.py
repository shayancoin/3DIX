"""
Constraint Solver for Post-Processing Layouts
Validates and adjusts generated layouts based on room type configurations.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

# Type definitions (matching TypeScript types)
@dataclass
class LayoutObject:
    id: str
    category: str
    position: Tuple[float, float, float]  # [x, y, z]
    size: Tuple[float, float, float]  # [width, height, depth]
    orientation: float  # degrees
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConstraintViolation:
    id: str
    constraint_type: str
    message: str
    severity: str
    metric_value: float
    threshold: float
    unit: Optional[str] = None
    normalized_violation: float = 0.0
    object_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "constraint_type": self.constraint_type,
            "message": self.message,
            "severity": self.severity,
            "metric_value": float(self.metric_value),
            "threshold": float(self.threshold),
            "unit": self.unit,
            "normalized_violation": float(self.normalized_violation),
            "object_ids": self.object_ids,
        }


@dataclass
class ConstraintValidation:
    satisfied: bool
    max_violation: float
    violations: List[ConstraintViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "max_violation": float(self.max_violation),
            "violations": [v.to_dict() for v in self.violations],
        }


class ConstraintSolver:
    """
    Solves layout constraints based on room type configurations.
    """
    TOLERANCE = 1e-3

    def __init__(self, room_config: Dict[str, Any]):
        """
        Initialize solver with room type configuration.

        Args:
            room_config: Room type configuration dictionary
        """
        self.room_config = room_config
        self.categories = {cat['id']: cat for cat in room_config.get('categories', [])}
        self.constraints = room_config.get('constraints', {})
        self.zones = room_config.get('zones', [])
        self.layout_rules = self.constraints.get('layoutRules', [])
    def _normalized_violation(self, metric_value: float, threshold: float, direction: str = "max") -> float:
        if threshold == 0 or threshold is None:
            return 1.0 if metric_value > 0 else 0.0
        if direction == "max":
            return max(0.0, (metric_value - threshold) / abs(threshold))
        return max(0.0, (threshold - metric_value) / abs(threshold))

    def _make_violation(
        self,
        *,
        violation_id: str,
        constraint_type: str,
        message: str,
        severity: str,
        metric_value: float,
        threshold: float,
        unit: Optional[str] = None,
        object_ids: Optional[List[str]] = None,
        direction: str = "max",
    ) -> ConstraintViolation:
        return ConstraintViolation(
            id=violation_id,
            constraint_type=constraint_type,
            message=message,
            severity=severity,
            metric_value=metric_value,
            threshold=threshold,
            unit=unit,
            normalized_violation=self._normalized_violation(metric_value, threshold, direction),
            object_ids=object_ids or [],
        )

    def _group_objects(self, objects: List[LayoutObject]) -> Dict[str, List[LayoutObject]]:
        objects_by_category: Dict[str, List[LayoutObject]] = {}
        for obj in objects:
            objects_by_category.setdefault(obj.category, []).append(obj)
        return objects_by_category

    def _validate_counts(self, cat_id: str, category_config: Dict[str, Any], count: int) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        min_count = category_config.get("minCount", 1 if category_config.get("required") else 0)
        max_count = category_config.get("maxCount")

        if count < min_count:
            violations.append(
                self._make_violation(
                    violation_id=f"count_min_{cat_id}",
                    constraint_type="count_min",
                    message=f"Category '{cat_id}' requires at least {min_count} object(s); found {count}.",
                    severity="error",
                    metric_value=float(count),
                    threshold=float(min_count),
                    unit="count",
                    object_ids=[],
                    direction="min",
                )
            )

        if max_count is not None and count > max_count:
            violations.append(
                self._make_violation(
                    violation_id=f"count_max_{cat_id}",
                    constraint_type="count_max",
                    message=f"Category '{cat_id}' allows at most {max_count} object(s); found {count}.",
                    severity="warning",
                    metric_value=float(count),
                    threshold=float(max_count),
                    unit="count",
                    object_ids=[],
                    direction="max",
                )
            )
        return violations

    def _validate_size(self, obj: LayoutObject, category_config: Dict[str, Any]) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        min_size = category_config.get("minSize")
        max_size = category_config.get("maxSize")

        if min_size:
            for i, dim in enumerate(["width", "height", "depth"]):
                if obj.size[i] < min_size[i]:
                    violations.append(
                        self._make_violation(
                            violation_id=f"size_min_{obj.id}_{dim}",
                            constraint_type="size_min",
                            message=f"{obj.category} '{obj.id}' {dim} is below minimum ({obj.size[i]:.2f}m < {min_size[i]:.2f}m)",
                            severity="error",
                            metric_value=float(obj.size[i]),
                            threshold=float(min_size[i]),
                            unit="m",
                            object_ids=[obj.id],
                            direction="min",
                        )
                    )

        if max_size:
            for i, dim in enumerate(["width", "height", "depth"]):
                if obj.size[i] > max_size[i]:
                    violations.append(
                        self._make_violation(
                            violation_id=f"size_max_{obj.id}_{dim}",
                            constraint_type="size_max",
                            message=f"{obj.category} '{obj.id}' {dim} exceeds maximum ({obj.size[i]:.2f}m > {max_size[i]:.2f}m)",
                            severity="warning",
                            metric_value=float(obj.size[i]),
                            threshold=float(max_size[i]),
                            unit="m",
                            object_ids=[obj.id],
                            direction="max",
                        )
                    )

        return violations

    def _validate_position(self, obj: LayoutObject, category_config: Dict[str, Any]) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        allowed_positions = category_config.get("allowedPositions", "any")
        room_width = self.room_config.get("defaultDimensions", {}).get("width", 4)
        room_length = self.room_config.get("defaultDimensions", {}).get("length", 3)

        x, _, z = obj.position
        width, _, depth = obj.size

        # Out of bounds
        overrun = max(
            0.0,
            -x,
            x + width - room_width,
            -z,
            z + depth - room_length,
        )
        if overrun > 0:
            violations.append(
                self._make_violation(
                    violation_id=f"position_bounds_{obj.id}",
                    constraint_type="position_bounds",
                    message=f"{obj.category} '{obj.id}' exceeds room bounds by {overrun:.2f}m",
                    severity="error",
                    metric_value=float(overrun),
                    threshold=float(max(room_width, room_length)),
                    unit="m",
                    object_ids=[obj.id],
                    direction="max",
                )
            )

        # Proximity to wall / center placement
        if allowed_positions == "wall":
            threshold = 0.1
            distances_to_walls = [
                x,
                room_width - (x + width),
                z,
                room_length - (z + depth),
            ]
            nearest_wall = min(distances_to_walls)
            if nearest_wall > threshold:
                violations.append(
                    self._make_violation(
                        violation_id=f"position_wall_{obj.id}",
                        constraint_type="position_wall",
                        message=f"{obj.category} '{obj.id}' should sit against a wall (nearest wall {nearest_wall:.2f}m, desired ≤ {threshold}m)",
                        severity="warning",
                        metric_value=float(nearest_wall),
                        threshold=float(threshold),
                        unit="m",
                        object_ids=[obj.id],
                        direction="max",
                    )
                )
        elif allowed_positions == "center":
            center_x = room_width / 2
            center_z = room_length / 2
            obj_center_x = x + width / 2
            obj_center_z = z + depth / 2
            distance_from_center = float(
                np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_z - center_z) ** 2)
            )
            allowed_radius = min(room_width, room_length) * 0.3
            if distance_from_center > allowed_radius:
                violations.append(
                    self._make_violation(
                        violation_id=f"position_center_{obj.id}",
                        constraint_type="position_center",
                        message=f"{obj.category} '{obj.id}' should be closer to the room center (distance {distance_from_center:.2f}m > {allowed_radius:.2f}m).",
                        severity="warning",
                        metric_value=distance_from_center,
                        threshold=float(allowed_radius),
                        unit="m",
                        object_ids=[obj.id],
                        direction="max",
                    )
                )

        return violations

    def _validate_spacing(self, obj: LayoutObject, peer_objects: List[LayoutObject], category_config: Dict[str, Any]) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        spacing = category_config.get("spacing", {})
        min_distance = spacing.get("minDistance")
        clearance = spacing.get("clearance", 0)

        if not min_distance and not clearance:
            return violations

        for other_obj in peer_objects:
            # Minimum distance between centers
            if min_distance:
                obj_center = np.array([obj.position[0] + obj.size[0] / 2, obj.position[2] + obj.size[2] / 2])
                other_center = np.array([other_obj.position[0] + other_obj.size[0] / 2, other_obj.position[2] + other_obj.size[2] / 2])
                distance = float(np.linalg.norm(obj_center - other_center))
                if distance < min_distance:
                    violations.append(
                        self._make_violation(
                            violation_id=f"spacing_min_{obj.id}_{other_obj.id}",
                            constraint_type="spacing_min_distance",
                            message=f"{obj.category} '{obj.id}' is too close to '{other_obj.id}' ({distance:.2f}m < {min_distance:.2f}m).",
                            severity="warning",
                            metric_value=distance,
                            threshold=float(min_distance),
                            unit="m",
                            object_ids=[obj.id, other_obj.id],
                            direction="min",
                        )
                    )

            # Clearance (axis-aligned gap)
            if clearance:
                x1_min, x1_max = obj.position[0], obj.position[0] + obj.size[0]
                z1_min, z1_max = obj.position[2], obj.position[2] + obj.size[2]

                x2_min, x2_max = other_obj.position[0], other_obj.position[0] + other_obj.size[0]
                z2_min, z2_max = other_obj.position[2], other_obj.position[2] + other_obj.size[2]

                gap_x = max(0.0, x2_min - x1_max, x1_min - x2_max)
                gap_z = max(0.0, z2_min - z1_max, z1_min - z2_max)
                actual_clearance = float(min(gap_x, gap_z)) if min(gap_x, gap_z) > 0 else 0.0

                if actual_clearance < clearance:
                    violations.append(
                        self._make_violation(
                            violation_id=f"spacing_clearance_{obj.id}_{other_obj.id}",
                            constraint_type="spacing_clearance",
                            message=f"{obj.category} '{obj.id}' lacks clearance around '{other_obj.id}' (clearance {actual_clearance:.2f}m < {clearance:.2f}m).",
                            severity="warning",
                            metric_value=actual_clearance,
                            threshold=float(clearance),
                            unit="m",
                            object_ids=[obj.id, other_obj.id],
                            direction="min",
                        )
                    )

        return violations

    def _validate_dependencies(self, objects_by_category: Dict[str, List[LayoutObject]]) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        for cat_id, category_config in self.categories.items():
            dependencies = category_config.get("dependencies", [])
            category_objects = objects_by_category.get(cat_id, [])

            if category_objects and dependencies:
                for dep_id in dependencies:
                    if dep_id not in objects_by_category or len(objects_by_category[dep_id]) == 0:
                        for obj in category_objects:
                            violations.append(
                                self._make_violation(
                                    violation_id=f"dependency_{cat_id}_requires_{dep_id}",
                                    constraint_type="dependency_missing",
                                    message=f"{obj.category} '{obj.id}' requires category '{dep_id}' to be present.",
                                    severity="error",
                                    metric_value=0.0,
                                    threshold=1.0,
                                    unit="count",
                                    object_ids=[obj.id],
                                    direction="min",
                                )
                            )
        return violations

    def _validate_conflicts(self, objects_by_category: Dict[str, List[LayoutObject]]) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        for cat_id, category_config in self.categories.items():
            conflicts = category_config.get("conflicts", [])
            category_objects = objects_by_category.get(cat_id, [])

            if category_objects and conflicts:
                for conflict_id in conflicts:
                    if conflict_id in objects_by_category and len(objects_by_category[conflict_id]) > 0:
                        for obj in category_objects:
                            violations.append(
                                self._make_violation(
                                    violation_id=f"conflict_{cat_id}_with_{conflict_id}_{obj.id}",
                                    constraint_type="conflict",
                                    message=f"{obj.category} '{obj.id}' conflicts with '{conflict_id}'.",
                                    severity="error",
                                    metric_value=1.0,
                                    threshold=0.0,
                                    unit="binary",
                                    object_ids=[obj.id],
                                    direction="max",
                                )
                            )
        return violations

    def _validate_layout_rule(self, rule: Dict[str, Any], objects: List[LayoutObject], objects_by_category: Dict[str, List[LayoutObject]]) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []

        rule_type = rule.get('type')
        category_ids = rule.get('categoryIds', [])
        parameters = rule.get('parameters', {})
        priority = rule.get('priority', 'optional')
        severity = "error" if priority == "required" else ("warning" if priority == "recommended" else "info")
        rule_id = rule.get('id', 'layout_rule')

        if rule_type == 'proximity':
            max_distance = parameters.get('maxDistance', 2.0)
            relevant_objects = [obj for obj in objects if obj.category in category_ids]

            if len(relevant_objects) < 2:
                return violations

            # Check pairwise distances
            for i, obj1 in enumerate(relevant_objects):
                for obj2 in relevant_objects[i+1:]:
                    obj1_center = np.array([obj1.position[0] + obj1.size[0] / 2, obj1.position[2] + obj1.size[2] / 2])
                    obj2_center = np.array([obj2.position[0] + obj2.size[0] / 2, obj2.position[2] + obj2.size[2] / 2])
                    distance = float(np.linalg.norm(obj1_center - obj2_center))

                    if distance > max_distance:
                        violations.append(
                            self._make_violation(
                                violation_id=f"{rule_id}_{obj1.id}_{obj2.id}",
                                constraint_type="layout_rule_proximity",
                                message=f"Rule '{rule.get('name', rule_id)}' violated: {obj1.category} and {obj2.category} are {distance:.2f}m apart (max {max_distance:.2f}m).",
                                severity=severity,
                                metric_value=distance,
                                threshold=float(max_distance),
                                unit="m",
                                object_ids=[obj1.id, obj2.id],
                                direction="max",
                            )
                        )

        elif rule_type == 'accessibility':
            min_clearance = parameters.get('minClearance', 0.5)
            relevant_objects = [obj for obj in objects if obj.category in category_ids]

            for obj in relevant_objects:
                # Check if object has clearance on at least one side
                room_width = self.room_config.get('defaultDimensions', {}).get('width', 4)
                room_length = self.room_config.get('defaultDimensions', {}).get('length', 3)

                has_clearance = (
                    obj.position[0] >= min_clearance or  # Left side
                    obj.position[0] + obj.size[0] <= room_width - min_clearance or  # Right side
                    obj.position[2] >= min_clearance or  # Back side
                    obj.position[2] + obj.size[2] <= room_length - min_clearance  # Front side
                )

                if not has_clearance:
                    violations.append(
                        self._make_violation(
                            violation_id=f"{rule_id}_{obj.id}",
                            constraint_type="layout_rule_accessibility",
                            message=f"Rule '{rule.get('name', rule_id)}' violated: {obj.category} '{obj.id}' lacks {min_clearance:.2f}m clearance on at least one side.",
                            severity=severity,
                            metric_value=0.0,
                            threshold=float(min_clearance),
                            unit="m",
                            object_ids=[obj.id],
                            direction="min",
                        )
                    )

        return violations

    def validate_layout(self, objects: List[LayoutObject]) -> ConstraintValidation:
        """
        Validate a layout against room type constraints and return normalized violations.
        """
        violations: List[ConstraintViolation] = []
        objects_by_category = self._group_objects(objects)

        # Required categories
        required_categories = self.constraints.get('requiredCategories', [])
        for cat_id in required_categories:
            count = len(objects_by_category.get(cat_id, []))
            if count < 1:
                violations.append(
                    self._make_violation(
                        violation_id=f"missing_required_{cat_id}",
                        constraint_type="missing_required",
                        message=f"Required category '{cat_id}' is missing.",
                        severity="error",
                        metric_value=float(count),
                        threshold=1.0,
                        unit="count",
                        object_ids=[],
                        direction="min",
                    )
                )

        # Category-level counts and per-object checks
        for cat_id, category_config in self.categories.items():
            category_objects = objects_by_category.get(cat_id, [])
            violations.extend(self._validate_counts(cat_id, category_config, len(category_objects)))

            for obj in category_objects:
                violations.extend(self._validate_size(obj, category_config))
                violations.extend(self._validate_position(obj, category_config))

        # Spacing checks (pairwise)
        for idx, obj in enumerate(objects):
            category_config = self.categories.get(obj.category, {})
            violations.extend(self._validate_spacing(obj, objects[idx + 1 :], category_config))

        # Dependencies and conflicts
        violations.extend(self._validate_dependencies(objects_by_category))
        violations.extend(self._validate_conflicts(objects_by_category))

        # Layout rules
        for rule in self.layout_rules:
            violations.extend(self._validate_layout_rule(rule, objects, objects_by_category))

        max_violation = max((v.normalized_violation for v in violations), default=0.0)
        satisfied = max_violation <= self.TOLERANCE

        return ConstraintValidation(
            satisfied=satisfied,
            max_violation=float(max_violation),
            violations=violations,
        )

    def solve_constraints(self, objects: List[LayoutObject]) -> Tuple[List[LayoutObject], ConstraintValidation]:
        """
        Solve constraints by adjusting object positions and sizes.

        Args:
            objects: List of layout objects to adjust

        Returns:
            Tuple of (adjusted_objects, validation_result)
        """
        # First validate to collect issues to fix
        validation = self.validate_layout(objects)

        # Create a copy to modify
        adjusted_objects = [
            LayoutObject(
                id=obj.id,
                category=obj.category,
                position=obj.position,
                size=obj.size,
                orientation=obj.orientation,
                metadata=obj.metadata
            )
            for obj in objects
        ]

        # Apply fixes for errors
        for violation in validation.violations:
            if violation.severity != "error" or violation.normalized_violation <= 0:
                continue

            if violation.constraint_type in ('position_bounds', 'position_wall', 'position_center'):
                obj_id = violation.object_ids[0] if violation.object_ids else None
                obj = next((o for o in adjusted_objects if o.id == obj_id), None)
                if obj:
                    category_config = self.categories.get(obj.category, {})
                    allowed_positions = category_config.get('allowedPositions', 'any')
                    room_width = self.room_config.get('defaultDimensions', {}).get('width', 4)
                    room_length = self.room_config.get('defaultDimensions', {}).get('length', 3)

                    if allowed_positions == 'wall':
                        # Move to nearest wall
                        x, y, z = obj.position
                        width, height, depth = obj.size

                        distances_to_walls = [
                            x,  # Left wall
                            room_width - (x + width),  # Right wall
                            z,  # Back wall
                            room_length - (z + depth),  # Front wall
                        ]
                        nearest_wall = distances_to_walls.index(min(distances_to_walls))

                        if nearest_wall == 0:  # Left
                            obj.position = (0.05, y, z)
                        elif nearest_wall == 1:  # Right
                            obj.position = (room_width - width - 0.05, y, z)
                        elif nearest_wall == 2:  # Back
                            obj.position = (x, y, 0.05)
                        else:  # Front
                            obj.position = (x, y, room_length - depth - 0.05)

            elif violation.constraint_type in ('size_min', 'size_max'):
                obj_id = violation.object_ids[0] if violation.object_ids else None
                obj = next((o for o in adjusted_objects if o.id == obj_id), None)
                if obj:
                    category_config = self.categories.get(obj.category, {})
                    min_size = category_config.get('minSize')
                    max_size = category_config.get('maxSize')

                    if min_size:
                        obj.size = tuple(max(obj.size[i], min_size[i]) for i in range(3))
                    if max_size:
                        obj.size = tuple(min(obj.size[i], max_size[i]) for i in range(3))

        # Re-validate after fixes
        final_validation = self.validate_layout(adjusted_objects)

        return adjusted_objects, final_validation
