"""Unit tests for ReflectanceParameters model"""

import pytest
from src.models.reflectance_parameters import ReflectanceParameters


class TestReflectanceParameters:
    """Tests for ReflectanceParameters class"""

    def test_reflectance_parameters_initialization(self):
        """Test ReflectanceParameters default initialization"""
        reflectance = ReflectanceParameters()
        # All reflectance values should exist with default None values
        assert reflectance.ceiling_reflectance is None
        assert reflectance.floor_reflectance is None
        assert reflectance.wall_reflectance is None
        assert reflectance.window_frame_reflectance is None

    def test_reflectance_parameters_with_values(self):
        """Test ReflectanceParameters with provided values"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=0.8,
            floor_reflectance=0.3,
            window_frame_reflectance=0.8
        )
        assert reflectance.ceiling_reflectance == 0.8
        assert reflectance.floor_reflectance == 0.3
        assert reflectance.window_frame_reflectance == 0.8

    def test_reflectance_parameters_to_dict(self):
        """Test converting ReflectanceParameters to dict"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=0.8,
            floor_reflectance=0.3,
            window_frame_reflectance=0.8
        )
        result = reflectance.to_dict()
        assert isinstance(result, dict)
        assert "ceiling_reflectance" in result
        assert result["ceiling_reflectance"] == 0.8
        assert "floor_reflectance" in result
        assert result["floor_reflectance"] == 0.3

    def test_reflectance_parameters_from_dict(self):
        """Test creating ReflectanceParameters from dict"""
        data = {
            "ceiling_reflectance": 0.8,
            "floor_reflectance": 0.3,
            "window_frame_reflectance": 0.8
        }
        reflectance = ReflectanceParameters.from_dict(data)
        assert reflectance.ceiling_reflectance == 0.8
        assert reflectance.floor_reflectance == 0.3
        assert reflectance.window_frame_reflectance == 0.8

    def test_reflectance_parameters_validate_valid_values(self):
        """Test that valid reflectance values pass validation"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=0.5,
            floor_reflectance=0.5,
            window_frame_reflectance=0.5
        )
        is_valid, message = reflectance.validate()
        assert is_valid is True
        assert message == ""

    def test_reflectance_parameters_validate_invalid_low(self):
        """Test validation fails for values below 0"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=-0.1,
            floor_reflectance=0.5
        )
        is_valid, message = reflectance.validate()
        assert is_valid is False
        assert "ceiling_reflectance" in message

    def test_reflectance_parameters_validate_invalid_high(self):
        """Test validation fails for values above 1"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=1.5,
            floor_reflectance=0.5
        )
        is_valid, message = reflectance.validate()
        assert is_valid is False
        assert "ceiling_reflectance" in message

    def test_reflectance_parameters_boundary_values(self):
        """Test boundary values 0 and 1"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=0.0,
            floor_reflectance=1.0,
            window_frame_reflectance=0.5
        )
        is_valid, message = reflectance.validate()
        assert is_valid is True
        assert reflectance.ceiling_reflectance == 0.0
        assert reflectance.floor_reflectance == 1.0

    def test_reflectance_parameters_partial_initialization(self):
        """Test partial initialization with some None values"""
        reflectance = ReflectanceParameters(
            ceiling_reflectance=0.8,
            floor_reflectance=None,
            wall_reflectance=0.6
        )
        assert reflectance.ceiling_reflectance == 0.8
        assert reflectance.floor_reflectance is None
        assert reflectance.wall_reflectance == 0.6
