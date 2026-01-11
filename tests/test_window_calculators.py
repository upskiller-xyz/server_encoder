"""
Unit tests for window parameter calculators

Tests all cases for window_sill_height and window_height calculations:
1. Normal case: window fully above floor
2. Edge case: window bottom exactly at floor level
3. Below floor case: window bottom below floor (requires adjustment)
"""

import unittest
from src.components.parameter_calculators import (
    WindowHeightCalculator,
    WindowSillHeightCalculator,
    ParameterCalculatorRegistry
)


class TestWindowHeightCalculator(unittest.TestCase):
    """Test cases for WindowHeightCalculator"""

    def setUp(self):
        """Set up calculator instance"""
        self.calculator = WindowHeightCalculator()

    def test_normal_case_window_above_floor(self):
        """Test normal case: window fully above floor"""
        parameters = {
            "z1": 1.0,  # bottom
            "z2": 3.0,  # top
            "floor_height_above_terrain": 0.5
        }

        # Window is above floor, so height = abs(z2 - z1) = 2.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.0)

    def test_window_bottom_at_floor_level(self):
        """Test edge case: window bottom exactly at floor level"""
        parameters = {
            "z1": 1.0,  # bottom at floor
            "z2": 3.0,  # top
            "floor_height_above_terrain": 1.0  # floor at window bottom
        }

        # Window bottom equals floor, so height = abs(z2 - z1) = 2.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.0)

    def test_window_bottom_below_floor(self):
        """Test case: window bottom below floor, calculate from floor"""
        parameters = {
            "z1": 0.5,  # bottom below floor
            "z2": 3.0,  # top
            "floor_height_above_terrain": 1.0
        }

        # Window bottom below floor, so height = max(z1,z2) - floor = 3.0 - 1.0 = 2.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.0)

    def test_window_bottom_significantly_below_floor(self):
        """Test case: window bottom significantly below floor"""
        parameters = {
            "z1": -1.0,  # bottom well below floor
            "z2": 2.5,   # top
            "floor_height_above_terrain": 0.5
        }

        # Height from floor: 2.5 - 0.5 = 2.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.0)

    def test_without_floor_height(self):
        """Test fallback: no floor_height_above_terrain provided"""
        parameters = {
            "z1": 1.0,
            "z2": 3.5
        }

        # No floor height, use full window height: abs(3.5 - 1.0) = 2.5
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.5)

    def test_with_window_geometry_dict(self):
        """Test using window_geometry dictionary format"""
        parameters = {
            "window_geometry": {
                "z1": 0.5,
                "z2": 3.0
            },
            "floor_height_above_terrain": 1.0
        }

        # Window bottom below floor: 3.0 - 1.0 = 2.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.0)

    def test_reversed_z_coordinates(self):
        """Test with z2 < z1 (reversed coordinates)"""
        parameters = {
            "z1": 3.0,  # top
            "z2": 0.5,  # bottom
            "floor_height_above_terrain": 1.0
        }

        # Window bottom (0.5) below floor (1.0): 3.0 - 1.0 = 2.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 2.0)

    def test_can_calculate_with_z_coords(self):
        """Test can_calculate returns True with z1, z2"""
        parameters = {"z1": 1.0, "z2": 2.0}
        self.assertTrue(self.calculator.can_calculate(parameters))

    def test_can_calculate_with_geometry(self):
        """Test can_calculate returns True with window_geometry"""
        parameters = {"window_geometry": {"z1": 1.0, "z2": 2.0}}
        self.assertTrue(self.calculator.can_calculate(parameters))

    def test_can_calculate_without_coords(self):
        """Test can_calculate returns False without coordinates"""
        parameters = {"floor_height_above_terrain": 1.0}
        self.assertFalse(self.calculator.can_calculate(parameters))


class TestWindowSillHeightCalculator(unittest.TestCase):
    """Test cases for WindowSillHeightCalculator"""

    def setUp(self):
        """Set up calculator instance"""
        self.calculator = WindowSillHeightCalculator()

    def test_normal_case_window_above_floor(self):
        """Test normal case: window fully above floor"""
        parameters = {
            "z1": 2.0,  # bottom
            "z2": 4.0,  # top
            "floor_height_above_terrain": 0.5
        }

        # Sill height = min(z1, z2) - floor = 2.0 - 0.5 = 1.5
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 1.5)

    def test_window_bottom_at_floor_level(self):
        """Test edge case: window bottom exactly at floor level"""
        parameters = {
            "z1": 1.0,  # bottom at floor
            "z2": 3.0,  # top
            "floor_height_above_terrain": 1.0
        }

        # Sill height = 1.0 - 1.0 = 0.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 0.0)

    def test_window_bottom_below_floor_capped_to_zero(self):
        """Test case: window bottom below floor, capped to 0"""
        parameters = {
            "z1": 0.5,  # bottom below floor
            "z2": 3.0,  # top
            "floor_height_above_terrain": 1.0
        }

        # Would be negative: 0.5 - 1.0 = -0.5, but capped to 0.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 0.0)

    def test_window_bottom_significantly_below_floor(self):
        """Test case: window bottom significantly below floor"""
        parameters = {
            "z1": -2.0,  # bottom well below floor
            "z2": 2.5,   # top
            "floor_height_above_terrain": 0.5
        }

        # Would be: -2.0 - 0.5 = -2.5, capped to 0.0
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 0.0)

    def test_with_window_geometry_dict(self):
        """Test using window_geometry dictionary format"""
        parameters = {
            "window_geometry": {
                "z1": 2.5,
                "z2": 4.0
            },
            "floor_height_above_terrain": 1.0
        }

        # Sill height = 2.5 - 1.0 = 1.5
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 1.5)

    def test_reversed_z_coordinates(self):
        """Test with z2 < z1 (reversed coordinates)"""
        parameters = {
            "z1": 4.0,  # top
            "z2": 2.0,  # bottom
            "floor_height_above_terrain": 0.5
        }

        # min(4.0, 2.0) - 0.5 = 2.0 - 0.5 = 1.5
        result = self.calculator.calculate(parameters)
        self.assertEqual(result, 1.5)

    def test_can_calculate_with_all_required(self):
        """Test can_calculate returns True with all required parameters"""
        parameters = {
            "z1": 1.0,
            "z2": 2.0,
            "floor_height_above_terrain": 0.5
        }
        self.assertTrue(self.calculator.can_calculate(parameters))

    def test_can_calculate_without_floor_height(self):
        """Test can_calculate returns False without floor_height_above_terrain"""
        parameters = {"z1": 1.0, "z2": 2.0}
        self.assertFalse(self.calculator.can_calculate(parameters))

    def test_can_calculate_without_z_coords(self):
        """Test can_calculate returns False without z coordinates"""
        parameters = {"floor_height_above_terrain": 1.0}
        self.assertFalse(self.calculator.can_calculate(parameters))


class TestParameterCalculatorRegistry(unittest.TestCase):
    """Test cases for ParameterCalculatorRegistry"""

    def test_calculates_both_parameters_normal_case(self):
        """Test registry calculates both window parameters - normal case"""
        parameters = {
            "z1": 2.0,
            "z2": 4.0,
            "floor_height_above_terrain": 0.5
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        # Should have both calculated
        self.assertIn("window_height", result)
        self.assertIn("window_sill_height", result)

        # window_height = 4.0 - 2.0 = 2.0
        self.assertEqual(result["window_height"], 2.0)

        # window_sill_height = 2.0 - 0.5 = 1.5
        self.assertEqual(result["window_sill_height"], 1.5)

    def test_calculates_with_window_below_floor(self):
        """Test registry with window bottom below floor"""
        parameters = {
            "z1": 0.5,   # below floor
            "z2": 3.0,
            "floor_height_above_terrain": 1.0
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        # window_height from floor: 3.0 - 1.0 = 2.0
        self.assertEqual(result["window_height"], 2.0)

        # window_sill_height capped: max(0, 0.5 - 1.0) = 0.0
        self.assertEqual(result["window_sill_height"], 0.0)

    def test_skips_if_parameter_already_provided(self):
        """Test registry skips calculation if user provides value"""
        parameters = {
            "z1": 2.0,
            "z2": 4.0,
            "floor_height_above_terrain": 0.5,
            "window_height": 99.9  # User override
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        # Should keep user-provided value
        self.assertEqual(result["window_height"], 99.9)

        # But still calculate sill height
        self.assertEqual(result["window_sill_height"], 1.5)

    def test_handles_missing_floor_height_gracefully(self):
        """Test registry when floor_height_above_terrain is missing"""
        parameters = {
            "z1": 2.0,
            "z2": 4.0
            # No floor_height_above_terrain
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        # window_height should still be calculated (doesn't require floor height)
        self.assertIn("window_height", result)
        self.assertEqual(result["window_height"], 2.0)

        # window_sill_height should NOT be calculated (requires floor height)
        self.assertNotIn("window_sill_height", result)

    def test_consistency_between_calculators(self):
        """Test consistency: window at floor should have sill_height=0 and adjusted height"""
        parameters = {
            "z1": 0.5,   # bottom below floor
            "z2": 2.5,   # top
            "floor_height_above_terrain": 1.0
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        # Sill height should be 0 (window starts at floor)
        self.assertEqual(result["window_sill_height"], 0.0)

        # Height should be from floor to top: 2.5 - 1.0 = 1.5
        self.assertEqual(result["window_height"], 1.5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_window_height_zero_height_window(self):
        """Test window with zero height"""
        calculator = WindowHeightCalculator()
        parameters = {
            "z1": 2.0,
            "z2": 2.0,
            "floor_height_above_terrain": 1.0
        }

        result = calculator.calculate(parameters)
        self.assertEqual(result, 0.0)

    def test_window_sill_large_positive_value(self):
        """Test window with large sill height"""
        calculator = WindowSillHeightCalculator()
        parameters = {
            "z1": 10.0,
            "z2": 12.0,
            "floor_height_above_terrain": 0.1
        }

        result = calculator.calculate(parameters)
        self.assertEqual(result, 9.9)

    def test_floor_height_zero(self):
        """Test with floor at terrain level (floor_height = 0)"""
        parameters = {
            "z1": 1.5,
            "z2": 3.0,
            "floor_height_above_terrain": 0.0
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        self.assertEqual(result["window_sill_height"], 1.5)
        self.assertEqual(result["window_height"], 1.5)

    def test_negative_floor_height(self):
        """Test with negative floor height (below terrain)"""
        parameters = {
            "z1": 1.0,
            "z2": 3.0,
            "floor_height_above_terrain": -0.5  # Below terrain
        }

        result = ParameterCalculatorRegistry.calculate_derived_parameters(parameters)

        # Sill height = 1.0 - (-0.5) = 1.5
        self.assertEqual(result["window_sill_height"], 1.5)
        # Normal height
        self.assertEqual(result["window_height"], 2.0)


if __name__ == '__main__':
    unittest.main()
