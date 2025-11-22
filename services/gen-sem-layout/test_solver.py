import unittest
from constraint_solver import ConstraintSolver, LayoutObject
from room_configs import ROOM_TYPE_CONFIGS


class TestConstraintSolver(unittest.TestCase):
    def setUp(self):
        self.bedroom_config = ROOM_TYPE_CONFIGS['bedroom']
        self.kitchen_config = ROOM_TYPE_CONFIGS['kitchen']
        self.solver = ConstraintSolver(self.bedroom_config)

    def test_validate_layout_valid(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(1.6, 0.5, 2.0), orientation=0),
            LayoutObject(id="ns1", category="nightstand", position=(1.1, 0, 0.5), size=(0.4, 0.5, 0.4), orientation=0),
        ]
        validation = self.solver.validate_layout(objects)
        self.assertTrue(validation.satisfied)
        self.assertEqual(len(validation.violations), 0)

    def test_validate_layout_missing_required(self):
        validation = self.solver.validate_layout([])
        self.assertFalse(validation.satisfied)
        self.assertTrue(any(v.constraint_type == 'missing_required' for v in validation.violations))

    def test_validate_layout_position_violation(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(1.6, 0.5, 2.0), orientation=0),
            LayoutObject(id="dresser1", category="dresser", position=(2.0, 0, 1.5), size=(1.0, 1.0, 0.5), orientation=0),
        ]
        validation = self.solver.validate_layout(objects)
        self.assertFalse(validation.satisfied)
        self.assertTrue(any(v.constraint_type.startswith('position_') for v in validation.violations))

    def test_solve_constraints_fixes_position(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(1.6, 0.5, 2.0), orientation=0),
            LayoutObject(id="dresser1", category="dresser", position=(2.0, 0, 1.5), size=(1.0, 1.0, 0.5), orientation=0),
        ]
        adjusted_objects, validation = self.solver.solve_constraints(objects)

        dresser = next(o for o in adjusted_objects if o.id == "dresser1")
        self.assertLessEqual(dresser.position[2], 0.1)  # moved toward wall
        self.assertTrue(validation.satisfied)

    def test_solve_constraints_fixes_size(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(0.5, 0.5, 0.5), orientation=0),
        ]
        adjusted_objects, validation = self.solver.solve_constraints(objects)

        bed = next(o for o in adjusted_objects if o.id == "bed1")
        self.assertGreaterEqual(bed.size[0], 1.4)  # Min width for bed
        self.assertTrue(validation.satisfied)

    def test_kitchen_triangle_violation_and_recovery(self):
        """A spread-out kitchen should report a work-triangle violation; a compact one should satisfy it."""
        solver = ConstraintSolver(self.kitchen_config)

        bad_objects = [
            LayoutObject(id="fridge", category="refrigerator", position=(0.0, 0, 0.0), size=(0.7, 1.8, 0.7), orientation=0),
            LayoutObject(id="sink", category="sink", position=(3.0, 0, 0.0), size=(0.6, 0.3, 0.6), orientation=0),
            LayoutObject(id="stove", category="stove", position=(3.0, 0, 2.5), size=(0.7, 1.0, 0.7), orientation=0),
            LayoutObject(id="counter", category="counter", position=(0.0, 0, 2.5), size=(0.6, 0.9, 0.6), orientation=0),
        ]
        bad_validation = solver.validate_layout(bad_objects)
        self.assertFalse(bad_validation.satisfied)
        self.assertTrue(any('kitchen_triangle' in v.id for v in bad_validation.violations))

        good_objects = [
            LayoutObject(id="fridge", category="refrigerator", position=(0.1, 0, 0.1), size=(0.7, 1.8, 0.7), orientation=0),
            LayoutObject(id="sink", category="sink", position=(1.5, 0, 0.1), size=(0.6, 0.3, 0.6), orientation=0),
            LayoutObject(id="stove", category="stove", position=(1.5, 0, 1.6), size=(0.7, 1.0, 0.7), orientation=0),
            LayoutObject(id="counter", category="counter", position=(0.1, 0, 1.6), size=(0.6, 0.9, 0.6), orientation=0),
        ]
        good_validation = solver.validate_layout(good_objects)
        self.assertTrue(good_validation.satisfied)
        self.assertLess(good_validation.max_violation, bad_validation.max_violation)


if __name__ == '__main__':
    unittest.main()
