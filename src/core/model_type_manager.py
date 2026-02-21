"""Model type management utility"""
import re


class ModelTypeManager:
    """Manager for model type operations"""

    @staticmethod
    def extract_prefix(model_type_str: str) -> str:
        """
        Extract model type prefix by removing version suffix.

        Supports model types with version suffixes like:
        - "df_default_2.0.1" -> "df_default"
        - "da_custom_1.5" -> "da_custom"
        - "df_default" -> "df_default" (unchanged)

        Args:
            model_type_str: Model type string possibly with version suffix

        Returns:
            Model type prefix without version suffix
        """
        # Match pattern: anything followed by underscore and version number (e.g., "_2.0.1", "_1.5")
        # Version pattern: _<digit(s)>.<digit(s)> optionally followed by .<digit(s)>
        version_pattern = r'_\d+\.\d+(?:\.\d+)?$'

        # Remove version suffix if present
        model_type_prefix = re.sub(version_pattern, '', model_type_str)

        return model_type_prefix
