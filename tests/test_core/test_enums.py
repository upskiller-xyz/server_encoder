"""Unit tests for core enums and constants"""

import pytest
from src.core.enums import (
    ModelType, ChannelType, FileFormat, RegionType,
    ParameterName, ImageDimensions, EncoderType,
    GeometryType, EncodingScheme
)


class TestModelType:
    """Tests for ModelType enum"""

    def test_model_types_exist(self):
        """Test that all expected model types are defined"""
        assert ModelType.DF_DEFAULT.value == "df_default"
        assert ModelType.DA_DEFAULT.value == "da_default"
        assert ModelType.DF_CUSTOM.value == "df_custom"
        assert ModelType.DA_CUSTOM.value == "da_custom"

    def test_model_type_by_value(self):
        """Test accessing model type by value"""
        assert ModelType("df_default") == ModelType.DF_DEFAULT
        assert ModelType("da_default") == ModelType.DA_DEFAULT


class TestChannelType:
    """Tests for ChannelType enum"""

    def test_channel_types_exist(self):
        """Test that all expected channel types are defined"""
        assert ChannelType.RED.value == "red"
        assert ChannelType.GREEN.value == "green"
        assert ChannelType.BLUE.value == "blue"
        assert ChannelType.ALPHA.value == "alpha"

    def test_channel_type_by_value(self):
        """Test accessing channel type by value"""
        assert ChannelType("red") == ChannelType.RED


class TestFileFormat:
    """Tests for FileFormat enum"""

    def test_file_formats_exist(self):
        """Test that all expected file formats are defined"""
        assert FileFormat.PNG.value == ".png"
        assert FileFormat.ARRAYS.value == "arrays"

    def test_file_format_by_value(self):
        """Test accessing file format by value"""
        assert FileFormat(".png") == FileFormat.PNG
        assert FileFormat("arrays") == FileFormat.ARRAYS


class TestRegionType:
    """Tests for RegionType enum"""

    def test_region_types_exist(self):
        """Test that all expected region types are defined"""
        assert RegionType.BACKGROUND.value == "background"
        assert RegionType.ROOM.value == "room"
        assert RegionType.WINDOW.value == "window"
        assert RegionType.OBSTRUCTION_BAR.value == "obstruction_bar"

    def test_region_type_by_value(self):
        """Test accessing region type by value"""
        assert RegionType("background") == RegionType.BACKGROUND


class TestParameterName:
    """Tests for ParameterName enum"""

    def test_parameter_names_exist(self):
        """Test that key parameter names are defined"""
        assert ParameterName.Z1.value == "z1"
        assert ParameterName.Z2.value == "z2"
        assert ParameterName.X1.value == "x1"
        assert ParameterName.X2.value == "x2"
        assert ParameterName.WINDOW_HEIGHT.value == "window_height"
        assert ParameterName.WINDOW_SILL_HEIGHT.value == "window_sill_height"
        assert ParameterName.ROOM_POLYGON.value == "room_polygon"
        assert ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value == "floor_height_above_terrain"

    def test_parameter_name_by_value(self):
        """Test accessing parameter name by value"""
        assert ParameterName("z1") == ParameterName.Z1
        assert ParameterName("room_polygon") == ParameterName.ROOM_POLYGON


class TestImageDimensions:
    """Tests for ImageDimensions class"""

    def test_image_dimensions_initialization(self):
        """Test ImageDimensions initialization"""
        dims = ImageDimensions(512)
        assert dims.image_size == 512

    def test_image_dimensions_obstruction_bar_position(self):
        """Test obstruction bar position calculation"""
        dims = ImageDimensions(512)
        x_start, y_start, x_end, y_end = dims.get_obstruction_bar_position()
        assert isinstance(x_start, int)
        assert isinstance(y_start, int)
        assert x_start >= 0
        assert y_start >= 0
        assert x_end > x_start
        assert y_end > y_start

    def test_obstruction_bar_height(self):
        """Test obstruction bar height property"""
        dims = ImageDimensions(512)
        bar_height = dims.obstruction_bar_height
        assert isinstance(bar_height, int)
        assert bar_height > 0

    def test_obstruction_bar_width(self):
        """Test obstruction bar width property"""
        dims = ImageDimensions(512)
        bar_width = dims.obstruction_bar_width
        assert isinstance(bar_width, int)
        assert bar_width > 0

    def test_different_image_widths(self):
        """Test dimensions with different image sizes"""
        for size in [256, 512, 1024]:
            dims = ImageDimensions(size)
            assert dims.image_size == size


class TestEncoderType:
    """Tests for EncoderType enum"""

    def test_encoder_types_exist(self):
        """Test that encoder types are defined"""
        assert EncoderType.LINEAR.value == "linear"
        assert EncoderType.ANGLE.value == "angle"
        assert EncoderType.REFLECTANCE.value == "reflectance"


class TestGeometryType:
    """Tests for GeometryType enum"""

    def test_geometry_types_exist(self):
        """Test that geometry types are defined"""
        assert GeometryType.POINT.value == "Point"
        assert GeometryType.POLYGON.value == "Polygon"
        assert GeometryType.LINE_STRING.value == "LineString"


class TestEncodingScheme:
    """Tests for EncodingScheme enum"""

    def test_encoding_schemes_exist(self):
        """Test that encoding schemes are defined"""
        assert EncodingScheme.RGB.value == "rgb"
        assert EncodingScheme.HSV.value == "hsv"

    def test_encoding_scheme_by_value(self):
        """Test accessing encoding scheme by value"""
        assert EncodingScheme("rgb") == EncodingScheme.RGB
        assert EncodingScheme("hsv") == EncodingScheme.HSV
