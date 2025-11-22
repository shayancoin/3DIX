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

    def __init__(self, room_config: Dict[str, Any]):
        """
        Create a ConstraintSolver configured for a specific room type.
        
        Initializes internal mappings and lists derived from the provided room configuration: category lookup by id, constraints dictionary, zones list, and layout rules list.
        
        Parameters:
            room_config (Dict[str, Any]): Room type configuration dictionary containing keys like 'categories', 'constraints', and 'zones'. The constructor extracts category configs into `self.categories`, constraint data into `self.constraints`, zone definitions into `self.zones`, and layout rules into `self.layout_rules`.
        """
        self.room_config = room_config
        self.categories = {cat['id']: cat for cat in room_config.get('categories', [])}
        self.constraints = room_config.get('constraints', {})
        self.zones = room_config.get('zones', [])
        self.layout_rules = self.constraints.get('layoutRules', [])

    def validate_layout(self, objects: List[LayoutObject]) -> ConstraintValidation:
        """
        Validate a layout against room type constraints.

        Args:
            objects: List of layout objects to validate

        Returns:
            ConstraintValidation with errors, warnings, and suggestions
        """
        errors: List[ConstraintError] = []
        warnings: List[ConstraintWarning] = []
        suggestions: List[ConstraintSuggestion] = []

        # Group objects by category
        objects_by_category: Dict[str, List[LayoutObject]] = {}
        for obj in objects:
            if obj.category not in objects_by_category:
                objects_by_category[obj.category] = []
            objects_by_category[obj.category].append(obj)

        # Check required categories
        required_categories = self.constraints.get('requiredCategories', [])
        for cat_id in required_categories:
            if cat_id not in objects_by_category or len(objects_by_category[cat_id]) == 0:
                errors.append(ConstraintError(
                    type='missing_required',
                    category_id=cat_id,
                    message=f"Required category '{cat_id}' is missing"
                ))

        # Validate each category
        for cat_id, category_config in self.categories.items():
            category_objects = objects_by_category.get(cat_id, [])

            # Check count constraints
            if category_config.get('required', False):
                if len(category_objects) < category_config.get('minCount', 1):
                    errors.append(ConstraintError(
                        type='count_violation',
                        category_id=cat_id,
                        message=f"Category '{cat_id}' requires at least {category_config.get('minCount', 1)} objects, found {len(category_objects)}"
                    ))

            max_count = category_config.get('maxCount')
            if max_count and len(category_objects) > max_count:
                errors.append(ConstraintError(
                    type='count_violation',
                    category_id=cat_id,
                    message=f"Category '{cat_id}' allows at most {max_count} objects, found {len(category_objects)}"
                ))

            # Validate each object in category
            for obj in category_objects:
                # Check size constraints
                size_errors = self._validate_size(obj, category_config)
                errors.extend(size_errors)

                # Check position constraints
                position_errors = self._validate_position(obj, category_config)
                errors.extend(position_errors)

                # Check spacing constraints
                spacing_warnings = self._validate_spacing(obj, objects, category_config)
                warnings.extend(spacing_warnings)

        # Check dependencies
        for cat_id, category_config in self.categories.items():
            dependencies = category_config.get("dependencies", [])
            category_objects = objects_by_category.get(cat_id, [])

            if category_objects and dependencies:
                for dep_id in dependencies:
                    if dep_id not in objects_by_category or len(objects_by_category[dep_id]) == 0:
                        for obj in category_objects:
                            errors.append(ConstraintError(
                                type='dependency_violation',
                                category_id=cat_id,
                                object_id=obj.id,
                                message=f"Object '{obj.id}' of category '{cat_id}' requires category '{dep_id}' to be present"
                            ))

        # Check conflicts
        for cat_id, category_config in self.categories.items():
            conflicts = category_config.get("conflicts", [])
            category_objects = objects_by_category.get(cat_id, [])

            if category_objects and conflicts:
                for conflict_id in conflicts:
                    if conflict_id in objects_by_category and len(objects_by_category[conflict_id]) > 0:
                        for obj in category_objects:
                            errors.append(ConstraintError(
                                type='conflict_violation',
                                category_id=cat_id,
                                object_id=obj.id,
                                message=f"Category '{cat_id}' conflicts with category '{conflict_id}'"
                            ))

        # Validate layout rules
        for rule in self.layout_rules:
            rule_warnings = self._validate_layout_rule(rule, objects, objects_by_category)
            warnings.extend(rule_warnings)

        # Generate suggestions
        suggestions = self._generate_suggestions(objects, objects_by_category, errors)
        suggestions.extend(suggestions)

        valid = len(errors) == 0
        return ConstraintValidation(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _validate_size(self, obj: LayoutObject, category_config: Dict[str, Any]) -> List[ConstraintError]:
        """
        Check whether an object's dimensions violate the category's minimum or maximum size constraints.
        
        Parameters:
            obj (LayoutObject): The layout object whose `size` (3-tuple) is validated.
            category_config (Dict[str, Any]): Category configuration that may contain `minSize` and `maxSize` as 3-element iterables.
        
        Returns:
            List[ConstraintError]: A list of `ConstraintError` entries with type `size_violation` for each detected size violation; empty if the object size is within the allowed bounds.
        """
        errors = []

        min_size = category_config.get('minSize')
        max_size = category_config.get('maxSize')

        if min_size:
            if any(obj.size[i] < min_size[i] for i in range(3)):
                errors.append(ConstraintError(
                    type='size_violation',
                    category_id=obj.category,
                    object_id=obj.id,
                    message=f"Object '{obj.id}' is smaller than minimum size for category '{obj.category}'"
                ))

        if max_size:
            if any(obj.size[i] > max_size[i] for i in range(3)):
                errors.append(ConstraintError(
                    type='size_violation',
                    category_id=obj.category,
                    object_id=obj.id,
                    message=f"Object '{obj.id}' is larger than maximum size for category '{obj.category}'"
                ))

        return errors

    def _validate_position(self, obj: LayoutObject, category_config: Dict[str, Any]) -> List[ConstraintError]:
        """
        Validate a layout object's position against room and category placement constraints.
        
        Parameters:
            obj (LayoutObject): The object whose position will be validated.
            category_config (Dict[str, Any]): Category-specific placement rules; may include `allowedPositions` among other placement settings.
        
        Returns:
            List[ConstraintError]: A list of position-related constraint errors. Each entry indicates a violation such as being outside room bounds, failing a required wall placement, or not being within the required center area.
        """
        errors = []

        allowed_positions = category_config.get('allowedPositions', 'any')
        room_width = self.room_config.get('defaultDimensions', {}).get('width', 4)
        room_length = self.room_config.get('defaultDimensions', {}).get('length', 3)

        x, y, z = obj.position
        width, height, depth = obj.size

        # Check if object is within room bounds
        if x < 0 or x + width > room_width or z < 0 or z + depth > room_length:
            errors.append(ConstraintError(
                type='position_violation',
                category_id=obj.category,
                object_id=obj.id,
                message=f"Object '{obj.id}' is outside room bounds"
            ))

        # Check position type constraints
        if allowed_positions == 'wall':
            # Object should be near a wall (within threshold)
            threshold = 0.1
            near_wall = (
                x <= threshold or x + width >= room_width - threshold or
                z <= threshold or z + depth >= room_length - threshold
            )
            if not near_wall:
                errors.append(ConstraintError(
                    type='position_violation',
                    category_id=obj.category,
                    object_id=obj.id,
                    message=f"Object '{obj.id}' must be placed against a wall"
                ))
        elif allowed_positions == 'center':
            # Object should be in center area
            center_x = room_width / 2
            center_z = room_length / 2
            obj_center_x = x + width / 2
            obj_center_z = z + depth / 2

            distance_from_center = np.sqrt(
                (obj_center_x - center_x) ** 2 + (obj_center_z - center_z) ** 2
            )
            if distance_from_center > min(room_width, room_length) * 0.3:
                errors.append(ConstraintError(
                    type='position_violation',
                    category_id=obj.category,
                    object_id=obj.id,
                    message=f"Object '{obj.id}' must be placed in center area"
                ))

        return errors

    def _validate_spacing(self, obj: LayoutObject, all_objects: List[LayoutObject], category_config: Dict[str, Any]) -> List[ConstraintWarning]:
        """
        Check spacing rules for a single object against all other objects and produce spacing-related warnings.
        
        Examines the category's `spacing` configuration for `minDistance` and `clearance`. If `minDistance` is set, emits a warning when another object's center-to-center distance is less than that value. If `clearance` is set, performs a simple 2D bounding-box overlap check (XZ plane) expanded by the clearance and emits a warning when boxes overlap.
        
        Parameters:
            obj (LayoutObject): The object to validate.
            all_objects (List[LayoutObject]): All layout objects to compare against (including `obj`).
            category_config (Dict[str, Any]): Category configuration which may contain a `spacing` dict with:
                - `minDistance` (float): Minimum required center-to-center distance in meters.
                - `clearance` (float): Required clearance in meters used to expand the object's bounding box.
        
        Returns:
            List[ConstraintWarning]: Warnings for spacing violations. Each warning has type `spacing_concern` and references the subject object's category and id.
        """
        warnings = []

        spacing = category_config.get('spacing', {})
        min_distance = spacing.get('minDistance')
        clearance = spacing.get('clearance', 0)

        if not min_distance and not clearance:
            return warnings

        for other_obj in all_objects:
            if other_obj.id == obj.id:
                continue

            # Calculate distance between object centers
            obj_center = np.array([obj.position[0] + obj.size[0] / 2, obj.position[2] + obj.size[2] / 2])
            other_center = np.array([other_obj.position[0] + other_obj.size[0] / 2, other_obj.position[2] + other_obj.size[2] / 2])
            distance = np.linalg.norm(obj_center - other_center)

            # Check minimum distance
            if min_distance and distance < min_distance:
                warnings.append(ConstraintWarning(
                    type='spacing_concern',
                    category_id=obj.category,
                    object_id=obj.id,
                    message=f"Object '{obj.id}' is too close to '{other_obj.id}' (distance: {distance:.2f}m, minimum: {min_distance}m)"
                ))

            # Check clearance (bounding box overlap)
            if clearance:
                # Simple bounding box check
                obj_bounds = {
                    'x_min': obj.position[0] - clearance,
                    'x_max': obj.position[0] + obj.size[0] + clearance,
                    'z_min': obj.position[2] - clearance,
                    'z_max': obj.position[2] + obj.size[2] + clearance,
                }
                other_bounds = {
                    'x_min': other_obj.position[0],
                    'x_max': other_obj.position[0] + other_obj.size[0],
                    'z_min': other_obj.position[2],
                    'z_max': other_obj.position[2] + other_obj.size[2],
                }

                if not (obj_bounds['x_max'] < other_bounds['x_min'] or
                       obj_bounds['x_min'] > other_bounds['x_max'] or
                       obj_bounds['z_max'] < other_bounds['z_min'] or
                       obj_bounds['z_min'] > other_bounds['z_max']):
                    warnings.append(ConstraintWarning(
                        type='spacing_concern',
                        category_id=obj.category,
                        object_id=obj.id,
                        message=f"Object '{obj.id}' does not have sufficient clearance around '{other_obj.id}'"
                    ))

        return warnings

    def _validate_layout_rule(self, rule: Dict[str, Any], objects: List[LayoutObject], objects_by_category: Dict[str, List[LayoutObject]]) -> List[ConstraintWarning]:
        """
        Validate a single layout rule against the provided layout objects and produce any resulting warnings.
        
        Parameters:
        	rule (Dict[str, Any]): Rule definition containing at least a `type` (e.g., "proximity" or "accessibility"), optional `categoryIds` (list of category ids to which the rule applies), and a `parameters` mapping with rule-specific values (for example, `maxDistance` for proximity or `minClearance` for accessibility).
        	objects (List[LayoutObject]): All layout objects to evaluate; used to select and compare relevant objects.
        	objects_by_category (Dict[str, List[LayoutObject]]): Mapping of category id to objects in that category (provided for convenience; the function may use `objects` directly).
        
        Returns:
        	List[ConstraintWarning]: Warnings describing rule violations (e.g., `suboptimal_layout` for objects farther apart than `maxDistance`, or `accessibility_issue` when an object lacks the required clearance).
        """
        warnings = []

        rule_type = rule.get('type')
        category_ids = rule.get('categoryIds', [])
        parameters = rule.get('parameters', {})
        priority = rule.get('priority', 'optional')

        if rule_type == 'proximity':
            max_distance = parameters.get('maxDistance', 2.0)
            relevant_objects = [obj for obj in objects if obj.category in category_ids]

            if len(relevant_objects) < 2:
                return warnings

            # Check pairwise distances
            for i, obj1 in enumerate(relevant_objects):
                for obj2 in relevant_objects[i+1:]:
                    obj1_center = np.array([obj1.position[0] + obj1.size[0] / 2, obj1.position[2] + obj1.size[2] / 2])
                    obj2_center = np.array([obj2.position[0] + obj2.size[0] / 2, obj2.position[2] + obj2.size[2] / 2])
                    distance = np.linalg.norm(obj1_center - obj2_center)

                    if distance > max_distance:
                        warnings.append(ConstraintWarning(
                            type='suboptimal_layout',
                            category_id=obj1.category,
                            message=f"Layout rule '{rule.get('name')}' violated: '{obj1.category}' and '{obj2.category}' are too far apart (distance: {distance:.2f}m, max: {max_distance}m)"
                        ))

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
                    warnings.append(ConstraintWarning(
                        type='accessibility_issue',
                        category_id=obj.category,
                        object_id=obj.id,
                        message=f"Object '{obj.id}' may not have sufficient accessibility clearance"
                    ))

        return warnings

    def _generate_suggestions(self, objects: List[LayoutObject], objects_by_category: Dict[str, List[LayoutObject]], errors: List[ConstraintError]) -> List[ConstraintSuggestion]:
        """
        Create suggestions to address validation issues, primarily by proposing additions for missing required categories.
        
        Parameters:
            objects (List[LayoutObject]): All layout objects currently in the room.
            objects_by_category (Dict[str, List[LayoutObject]]): Mapping from category id to list of objects of that category.
            errors (List[ConstraintError]): Validation errors detected (not directly used for current suggestions but provided for extensibility).
        
        Returns:
            List[ConstraintSuggestion]: A list of suggestions. Each suggestion of type `add_object` includes `category_id`, a `message`, and a `suggested_action` dictionary with `position`, `size`, and `orientation` for the recommended new object.
        """
        suggestions = []

        # Suggest adding missing required objects
        required_categories = self.constraints.get('requiredCategories', [])
        for cat_id in required_categories:
            if cat_id not in objects_by_category or len(objects_by_category[cat_id]) == 0:
                category_config = self.categories.get(cat_id, {})
                default_size = category_config.get('minSize', [1.0, 1.0, 1.0])
                room_width = self.room_config.get('defaultDimensions', {}).get('width', 4)
                room_length = self.room_config.get('defaultDimensions', {}).get('length', 3)

                # Suggest position based on allowed positions
                allowed_positions = category_config.get('allowedPositions', 'any')
                if allowed_positions == 'wall':
                    suggested_position = [0.1, 0, 0.1]  # Near corner
                elif allowed_positions == 'center':
                    suggested_position = [room_width / 2 - default_size[0] / 2, 0, room_length / 2 - default_size[2] / 2]
                else:
                    suggested_position = [room_width / 2 - default_size[0] / 2, 0, room_length / 2 - default_size[2] / 2]

                suggestions.append(ConstraintSuggestion(
                    type='add_object',
                    category_id=cat_id,
                    message=f"Add required object of category '{cat_id}'",
                    suggested_action={
                        'position': suggested_position,
                        'size': default_size,
                        'orientation': 0.0
                    }
                ))

        return suggestions

    def solve_constraints(self, objects: List[LayoutObject]) -> Tuple[List[LayoutObject], ConstraintValidation]:
        """
        Adjust layout objects to satisfy detected constraints and return the adjusted set with validation results.
        
        This method validates the provided objects, applies automated fixes for detectable errors (e.g., nudging objects against a wall when a position violation occurs for categories restricted to wall placement; clamping object sizes to category min/max when a size violation occurs), re-validates the adjusted objects, and returns both the modified object list and the final ConstraintValidation.
        
        Parameters:
            objects (List[LayoutObject]): The input layout objects to validate and potentially adjust.
        
        Returns:
            Tuple[List[LayoutObject], ConstraintValidation]: A tuple where the first element is the list of adjusted LayoutObject instances and the second is the validation result after applying fixes.
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

            elif error.type == 'size_violation':
                # Clamp size to valid range
                obj = next((o for o in adjusted_objects if o.id == error.object_id), None)
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
