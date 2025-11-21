import unittest
from constraint_solver import ConstraintSolver, LayoutObject
from room_configs import ROOM_TYPE_CONFIGS

class TestConstraintSolver(unittest.TestCase):
    def setUp(self):
        self.room_config = ROOM_TYPE_CONFIGS['bedroom']
        self.solver = ConstraintSolver(self.room_config)

    def test_validate_layout_valid(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(1.6, 0.5, 2.0), orientation=0),
            LayoutObject(id="ns1", category="nightstand", position=(1.1, 0, 0.5), size=(0.4, 0.5, 0.4), orientation=0)
        ]
        validation = self.solver.validate_layout(objects)
        self.assertTrue(validation.valid)
        self.assertEqual(len(validation.errors), 0)

    def test_validate_layout_missing_required(self):
        objects = []
        validation = self.solver.validate_layout(objects)
        self.assertFalse(validation.valid)
        self.assertTrue(any(e.type == 'missing_required' for e in validation.errors))

    def test_validate_layout_position_violation(self):
        # Dresser must be against wall
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(1.6, 0.5, 2.0), orientation=0),
            LayoutObject(id="dresser1", category="dresser", position=(2.0, 0, 1.5), size=(1.0, 1.0, 0.5), orientation=0) # In middle of room
        ]
        validation = self.solver.validate_layout(objects)
        self.assertFalse(validation.valid)
        self.assertTrue(any(e.type == 'position_violation' for e in validation.errors))

    def test_solve_constraints_fixes_position(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(1.6, 0.5, 2.0), orientation=0),
            LayoutObject(id="dresser1", category="dresser", position=(2.0, 0, 1.5), size=(1.0, 1.0, 0.5), orientation=0) # In middle
        ]
        adjusted_objects, validation = self.solver.solve_constraints(objects)

        # Check if dresser was moved to wall
        dresser = next(o for o in adjusted_objects if o.id == "dresser1")
        # Room is 4x3.5. Dresser at (2, 1.5) is far from walls.
        # Solver should move it to nearest wall.
        # Nearest wall to (2, 1.5) in 4x3.5 room:
        # x=0 (dist 2), x=4 (dist 2)
        # z=0 (dist 1.5), z=3.5 (dist 2)
        # Should move to z=0 or similar.

        # Let's check if it's valid now
        self.assertTrue(validation.valid)

    def test_solve_constraints_fixes_size(self):
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(0.5, 0.5, 0.5), orientation=0), # Too small
        ]
        adjusted_objects, validation = self.solver.solve_constraints(objects)

        bed = next(o for o in adjusted_objects if o.id == "bed1")
        self.assertGreaterEqual(bed.size[0], 1.4) # Min width for bed

if __name__ == '__main__':
    unittest.main()
