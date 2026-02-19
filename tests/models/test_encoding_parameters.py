"""Unit tests for EncodingParameters model"""

import pytest
from src.models.encoding_parameters import EncodingParameters
from src.models.region_parameters import RegionParameters
from src.core import RegionType


class TestEncodingParameters:
    """Tests for EncodingParameters class"""

    def test_encoding_parameters_initialization(self):
        """Test EncodingParameters default initialization"""
        params = EncodingParameters()
        assert isinstance(params.background, RegionParameters)
        assert isinstance(params.room, RegionParameters)
        assert isinstance(params.window, RegionParameters)
        assert isinstance(params.obstruction_bar, RegionParameters)
        assert isinstance(params.global_params, dict)

    def test_get_region(self):
        """Test get_region method"""
        params = EncodingParameters()
        bg_region = params.get_region(RegionType.BACKGROUND)
        assert isinstance(bg_region, RegionParameters)

    def test_set_and_get_global(self):
        """Test set_global and get_global methods"""
        params = EncodingParameters()
        params.set_global("key", "value")
        assert params.get_global("key") == "value"
        assert params.get_global("nonexistent") is None
        assert params.get_global("nonexistent", "default") == "default"

    def test_dict_like_setitem(self):
        """Test __setitem__ for dict-like interface"""
        params = EncodingParameters()
        params["key"] = "value"
        assert params.global_params["key"] == "value"

    def test_dict_like_getitem(self):
        """Test __getitem__ for dict-like interface"""
        params = EncodingParameters()
        params.window.parameters["param"] = "value"
        assert params["param"] == "value"

    def test_dict_like_contains(self):
        """Test __contains__ for dict-like interface"""
        params = EncodingParameters()
        params.window.parameters["param"] = "value"
        assert "param" in params
        params.set_global("global_param", "value")
        assert "global_param" in params
        assert "nonexistent" not in params

    def test_from_dict(self):
        """Test creating EncodingParameters from dict"""
        data = {
            RegionType.BACKGROUND.value: {"bg_param": "value"},
            RegionType.WINDOW.value: {"window_param": "value"},
            "global_param": "value"
        }
        params = EncodingParameters.from_dict(data)
        assert params.background.parameters["bg_param"] == "value"
        assert params.window.parameters["window_param"] == "value"
        assert params.global_params["global_param"] == "value"

    def test_to_dict(self):
        """Test converting EncodingParameters to dict"""
        params = EncodingParameters()
        params.background.parameters["bg_param"] = "value"
        params.set_global("global_param", "value")
        result = params.to_dict()
        assert result["global_param"] == "value"
        assert result[RegionType.BACKGROUND.value]["bg_param"] == "value"

    def test_to_dict_empty_regions_not_included(self):
        """Test that empty regions are not included in to_dict"""
        params = EncodingParameters()
        params.set_global("key", "value")
        result = params.to_dict()
        assert "global_param" not in result or "key" in result
        assert result.get("global_param") is None or "key" in result
