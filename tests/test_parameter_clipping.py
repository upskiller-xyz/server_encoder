"""
Unit tests for parameter clipping functionality

Tests clipping behavior for all parameters with clipping rules:
- floor_height_above_terrain: reject < 0, clip > 10
- height_roof_over_floor: reject <= 0, clip < 12 to 12, clip > 30 to 30
- obstruction_angle_horizon: clip to [0, 90]
- obstruction_angle_zenith: clip to [0, 70]
"""

import unittest
from unittest.mock import Mock
from src.components.encoding_service import EncodingService
from src.components.enums import ModelType


class TestFloorHeightAboveTerrainClipping(unittest.TestCase):
    """Test clipping for floor_height_above_terrain parameter"""

    def setUp(self):
        """Set up encoding service with mock logger"""
        self.logger = Mock()
        self.service = EncodingService(self.logger)

    def test_floor_height_negative_rejected(self):
        """Test negative floor height is rejected with error"""
        parameters = {
            "floor_height_above_terrain": -1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 1.0,
            "z2": 3.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertFalse(is_valid)
        self.assertIn("floor_height_above_terrain", error_msg)
        self.assertIn("not supported", error_msg)
        self.assertIn("-1.0", error_msg)

    def test_floor_height_zero_accepted(self):
        """Test floor height of 0 is accepted"""
        parameters = {
            "floor_height_above_terrain": 0.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 1.0,
            "z2": 3.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["floor_height_above_terrain"], 0.0)

    def test_floor_height_within_range_accepted(self):
        """Test floor height within range [0, 10] is accepted"""
        parameters = {
            "floor_height_above_terrain": 5.5,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 6.0,
            "z2": 8.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["floor_height_above_terrain"], 5.5)

    def test_floor_height_at_max_accepted(self):
        """Test floor height at max (10.0) is accepted"""
        parameters = {
            "floor_height_above_terrain": 10.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 11.0,
            "z2": 13.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["floor_height_above_terrain"], 10.0)

    def test_floor_height_above_max_clipped(self):
        """Test floor height > 10 is clipped to 10 with warning"""
        parameters = {
            "floor_height_above_terrain": 15.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 16.0,
            "z2": 18.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["floor_height_above_terrain"], 10.0)
        # Check warning was logged
        self.logger.warning.assert_called()


class TestHeightRoofOverFloorClipping(unittest.TestCase):
    """Test clipping for height_roof_over_floor parameter"""

    def setUp(self):
        """Set up encoding service with mock logger"""
        self.logger = Mock()
        self.service = EncodingService(self.logger)

    def test_height_negative_rejected(self):
        """Test negative height is rejected with error"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": -2.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertFalse(is_valid)
        self.assertIn("height_roof_over_floor", error_msg)
        self.assertIn("not supported", error_msg)

    def test_height_zero_rejected(self):
        """Test height of 0 is rejected (must be > 0)"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 0.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertFalse(is_valid)
        self.assertIn("height_roof_over_floor", error_msg)

    def test_height_small_positive_accepted(self):
        """Test small positive height is clipped to minimum of 15"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 0.1,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["height_roof_over_floor"], 15.0)

    def test_height_within_range_accepted(self):
        """Test height within range (0, 30] is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 15.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["height_roof_over_floor"], 15.0)

    def test_height_at_max_accepted(self):
        """Test height at max (30.0) is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 30.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["height_roof_over_floor"], 30.0)

    def test_height_above_max_clipped(self):
        """Test height > 30 is clipped to 30 with warning"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 50.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["height_roof_over_floor"], 30.0)
        # Check warning was logged
        self.logger.warning.assert_called()

    def test_height_significantly_above_max_clipped(self):
        """Test height >> 30 is clipped to 30"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 100.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["height_roof_over_floor"], 30.0)


class TestObstructionAngleHorizonClipping(unittest.TestCase):
    """Test clipping for obstruction_angle_horizon parameter"""

    def setUp(self):
        """Set up encoding service with mock logger"""
        self.logger = Mock()
        self.service = EncodingService(self.logger)

    def test_horizon_angle_negative_clipped_to_zero(self):
        """Test negative horizon angle is clipped to 0"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": -15.0,
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_horizon"], 0.0)

    def test_horizon_angle_zero_accepted(self):
        """Test horizon angle of 0 is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": 0.0,
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_horizon"], 0.0)

    def test_horizon_angle_within_range_accepted(self):
        """Test horizon angle within [0, 90] is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": 45.0,
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_horizon"], 45.0)

    def test_horizon_angle_at_max_accepted(self):
        """Test horizon angle at max (90) is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": 90.0,
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_horizon"], 90.0)

    def test_horizon_angle_above_max_clipped(self):
        """Test horizon angle > 90 is clipped to 90"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": 120.0,
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_horizon"], 90.0)


class TestObstructionAngleZenithClipping(unittest.TestCase):
    """Test clipping for obstruction_angle_zenith parameter"""

    def setUp(self):
        """Set up encoding service with mock logger"""
        self.logger = Mock()
        self.service = EncodingService(self.logger)

    def test_zenith_angle_negative_clipped_to_zero(self):
        """Test negative zenith angle is clipped to 0"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": -10.0,
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_zenith"], 0.0)

    def test_zenith_angle_zero_accepted(self):
        """Test zenith angle of 0 is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": 0.0,
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_zenith"], 0.0)

    def test_zenith_angle_within_range_accepted(self):
        """Test zenith angle within [0, 70] is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": 35.0,
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_zenith"], 35.0)

    def test_zenith_angle_at_max_accepted(self):
        """Test zenith angle at max (70) is accepted"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": 70.0,
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_zenith"], 70.0)

    def test_zenith_angle_above_max_clipped(self):
        """Test zenith angle > 70 is clipped to 70"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": 85.0,
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_zenith"], 70.0)

    def test_zenith_angle_significantly_above_max_clipped(self):
        """Test zenith angle >> 70 is clipped to 70"""
        parameters = {
            "floor_height_above_terrain": 1.0,
            "height_roof_over_floor": 3.0,
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": 180.0,
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["obstruction_angle_zenith"], 70.0)


class TestMultipleParametersClipping(unittest.TestCase):
    """Test clipping when multiple parameters need adjustment"""

    def setUp(self):
        """Set up encoding service with mock logger"""
        self.logger = Mock()
        self.service = EncodingService(self.logger)

    def test_multiple_parameters_clipped_simultaneously(self):
        """Test multiple parameters can be clipped in same validation"""
        parameters = {
            "floor_height_above_terrain": 15.0,     # Will clip to 10
            "height_roof_over_floor": 50.0,         # Will clip to 30
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": 120.0,     # Will clip to 90
            "obstruction_angle_zenith": 85.0,       # Will clip to 70
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 16.0,
            "z2": 18.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        self.assertTrue(is_valid)
        self.assertEqual(parameters["floor_height_above_terrain"], 10.0)
        self.assertEqual(parameters["height_roof_over_floor"], 30.0)
        self.assertEqual(parameters["obstruction_angle_horizon"], 90.0)
        self.assertEqual(parameters["obstruction_angle_zenith"], 70.0)

    def test_some_parameters_rejected_some_clipped(self):
        """Test that rejection takes precedence over clipping"""
        parameters = {
            "floor_height_above_terrain": -1.0,     # Will be rejected
            "height_roof_over_floor": 50.0,         # Would clip but doesn't matter
            "window_frame_ratio": 0.8,
            "obstruction_angle_horizon": [45.0],
            "obstruction_angle_zenith": [30.0],
            "room_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "z1": 2.0,
            "z2": 4.0
        }

        is_valid, error_msg = self.service.validate_parameters(parameters, ModelType.DF_DEFAULT)

        # Should fail validation due to negative floor_height
        self.assertFalse(is_valid)
        self.assertIn("floor_height_above_terrain", error_msg)


if __name__ == '__main__':
    unittest.main()
