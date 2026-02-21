"""Tests for ModelTypeManager utility class"""
import pytest
from src.core.model_type_manager import ModelTypeManager


class TestModelTypeManager:
    """Test suite for ModelTypeManager"""

    def test_extract_prefix_without_version(self):
        """Test extraction when no version suffix exists"""
        result = ModelTypeManager.extract_prefix("df_default")
        assert result == "df_default"

    def test_extract_prefix_with_simple_version(self):
        """Test extraction with simple two-part version (e.g., 1.5)"""
        result = ModelTypeManager.extract_prefix("df_default_1.5")
        assert result == "df_default"

    def test_extract_prefix_with_full_version(self):
        """Test extraction with full three-part version (e.g., 2.0.1)"""
        result = ModelTypeManager.extract_prefix("df_default_2.0.1")
        assert result == "df_default"

    def test_extract_prefix_da_custom(self):
        """Test extraction with da_custom model"""
        result = ModelTypeManager.extract_prefix("da_custom_1.5")
        assert result == "da_custom"

    def test_extract_prefix_da_custom_full_version(self):
        """Test extraction with da_custom and full version"""
        result = ModelTypeManager.extract_prefix("da_custom_2.1.3")
        assert result == "da_custom"

    def test_extract_prefix_df_custom(self):
        """Test extraction with df_custom model"""
        result = ModelTypeManager.extract_prefix("df_custom_1.0.0")
        assert result == "df_custom"

    def test_extract_prefix_da_default(self):
        """Test extraction with da_default model"""
        result = ModelTypeManager.extract_prefix("da_default_3.2.1")
        assert result == "da_default"

    def test_extract_prefix_preserves_underscores_in_name(self):
        """Test that underscores in model name are preserved"""
        result = ModelTypeManager.extract_prefix("complex_model_name_1.0")
        assert result == "complex_model_name"

    def test_extract_prefix_with_zero_version(self):
        """Test extraction with version starting at 0"""
        result = ModelTypeManager.extract_prefix("model_0.0.1")
        assert result == "model"

    def test_extract_prefix_single_digit_versions(self):
        """Test extraction with single digit versions"""
        result = ModelTypeManager.extract_prefix("model_5.9.3")
        assert result == "model"
