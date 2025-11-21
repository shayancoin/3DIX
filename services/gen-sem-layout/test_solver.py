import unittest
from constraint_solver import ConstraintSolver, LayoutObject
from room_configs import ROOM_TYPE_CONFIGS

class TestConstraintSolver(unittest.TestCase):
    def setUp(self):
        """
        Prepare the test fixture by loading the bedroom room configuration and creating a ConstraintSolver initialized with that configuration.
        
        Sets:
            self.room_config: The bedroom configuration dictionary from ROOM_TYPE_CONFIGS.
            self.solver: An instance of ConstraintSolver initialized with `self.room_config`.
        """
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
        """
        Verifies that solve_constraints repositions a dresser placed away from walls toward the nearest wall and produces a valid layout.
        
        Sets up a bed and a dresser positioned in the room center, runs the constraint solver, and asserts that the resulting validation marks the layout as valid. The test implies the dresser should be moved from a middle position to a location adjacent to a room wall.
        """
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
        """
        Verifies that the constraint solver increases an undersized bed to meet the minimum required width.
        
        Creates a bed with dimensions below the minimum, runs `solve_constraints`, and asserts the adjusted bed's width is at least 1.4 meters.
        """
        objects = [
            LayoutObject(id="bed1", category="bed", position=(1.5, 0, 0.5), size=(0.5, 0.5, 0.5), orientation=0), # Too small
        ]
        adjusted_objects, validation = self.solver.solve_constraints(objects)

        bed = next(o for o in adjusted_objects if o.id == "bed1")
        self.assertGreaterEqual(bed.size[0], 1.4) # Min width for bed

if __name__ == '__main__':
    unittest.main()