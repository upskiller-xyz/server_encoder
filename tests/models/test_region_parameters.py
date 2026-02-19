"""Unit tests for RegionParameters model"""

import pytest
from src.models.region_parameters import RegionParameters


class TestRegionParameters:
    """Tests for RegionParameters class"""

    def test_region_parameters_initialization(self):
        """Test RegionParameters default initialization"""
        params = RegionParameters()
        assert isinstance(params.parameters, dict)
        assert len(params.parameters) == 0

    def test_region_parameters_with_initial_data(self):
        """Test RegionParameters initialization with data"""
        data = {"param1": "value1", "param2": 42}
        params = RegionParameters(parameters=data)
        assert params.parameters == data

    def test_region_parameters_get(self):
        """Test get method"""
        params = RegionParameters(parameters={"key": "value"})
        assert params.get("key") == "value"
        assert params.get("nonexistent") is None
        assert params.get("nonexistent", "default") == "default"

    def test_region_parameters_setitem(self):
        """Test setting parameter via __setitem__"""
        params = RegionParameters()
        params["key"] = "value"
        assert params.parameters["key"] == "value"

    def test_region_parameters_getitem(self):
        """Test getting parameter via __getitem__"""
        params = RegionParameters(parameters={"key": "value"})
        assert params["key"] == "value"

    def test_region_parameters_contains(self):
        """Test __contains__ method"""
        params = RegionParameters(parameters={"key": "value"})
        assert "key" in params
        assert "nonexistent" not in params

    def test_region_parameters_update(self):
        """Test update method"""
        params = RegionParameters(parameters={"key1": "value1"})
        params.update({"key2": "value2"})
        assert params.parameters["key1"] == "value1"
        assert params.parameters["key2"] == "value2"

    def test_region_parameters_keys(self):
        """Test keys method"""
        params = RegionParameters(parameters={"key1": "value1", "key2": "value2"})
        assert set(params.keys()) == {"key1", "key2"}

    def test_region_parameters_values(self):
        """Test values method"""
        params = RegionParameters(parameters={"key1": "value1", "key2": "value2"})
        assert set(params.values()) == {"value1", "value2"}

    def test_region_parameters_items(self):
        """Test items method"""
        params = RegionParameters(parameters={"key1": "value1", "key2": "value2"})
        assert len(list(params.items())) == 2
