from typing import Dict, Any, Optional
import numpy as np
from src.components.interfaces import IImageBuilder
from src.components.enums import ModelType, RegionType, ParameterName, PARAMETER_REGIONS
from src.components.region_encoders import RegionEncoderFactory


class RoomImageBuilder(IImageBuilder):
    """
    Builder for creating encoded room images (Builder Pattern)

    Constructs 128×128 RGBA images with encoded room parameters
    """

    def __init__(self):
        self._image: Optional[np.ndarray] = None
        self._model_type: Optional[ModelType] = None
        self._region_encoder_factory = RegionEncoderFactory()
        self.reset()

    def reset(self) -> 'RoomImageBuilder':
        """Reset builder to initial state"""
        # Create 128×128 RGBA image initialized to zeros
        self._image = np.zeros((128, 128, 4), dtype=np.uint8)
        self._model_type = None
        return self

    def set_model_type(self, model_type: ModelType) -> 'RoomImageBuilder':
        """Set the model type for encoding"""
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Invalid model type: {model_type}")
        self._model_type = model_type
        return self

    def encode_region(self, region_type: RegionType, parameters: Dict[str, Any]) -> 'RoomImageBuilder':
        """
        Encode a region using its encoder (Single Responsibility Principle)

        Args:
            region_type: Type of region to encode
            parameters: Region-specific parameters

        Returns:
            Self for chaining
        """
        self._validate_state()
        encoder = self._region_encoder_factory.get_encoder(region_type)
        self._image = encoder.encode_region(self._image, parameters, self._model_type)
        return self

    def build(self) -> np.ndarray:
        """
        Build and return the final encoded image

        Returns:
            128×128 RGBA numpy array

        Raises:
            RuntimeError: If model type not set
        """
        self._validate_state()
        if self._image is None:
            raise RuntimeError("Image not initialized")
        return self._image.copy()

    def _validate_state(self) -> None:
        """Validate builder state before operations"""
        if self._model_type is None:
            raise RuntimeError("Model type must be set before encoding")
        if self._image is None:
            raise RuntimeError("Image not initialized. Call reset() first.")


class RoomImageDirector:
    """
    Director class for orchestrating image building (Director Pattern)

    Provides high-level interface for building complete encoded images
    """

    def __init__(self, builder: IImageBuilder):
        self._builder = builder

    def construct_full_image(
        self,
        model_type: ModelType,
        all_parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Construct a complete encoded image with all regions

        Args:
            model_type: The model type to use
            all_parameters: All parameters grouped by region

        Returns:
            Complete encoded image
        """
        # Reset and configure builder
        self._builder.reset().set_model_type(model_type)

        # Rotate geometry if window is not on south facade
        all_parameters = self._rotate_geometry_if_needed(all_parameters)

        # Define region encoding order (list-based iteration)
        region_order = [
            RegionType.BACKGROUND,
            RegionType.ROOM,
            RegionType.WINDOW,
            RegionType.OBSTRUCTION_BAR
        ]

        # Encode regions in order using list comprehension
        [self._builder.encode_region(region, all_parameters[region.value])
         for region in region_order
         if region.value in all_parameters]

        # Build final image
        return self._builder.build()

    def _rotate_geometry_if_needed(self, all_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rotate room polygon and window coordinates if window is not on south facade

        Args:
            all_parameters: Parameters grouped by region

        Returns:
            Parameters with rotated geometry
        """
        from src.components.geometry import WindowGeometry, RoomPolygon, Point2D
        from src.components.enums import ParameterName, FACADE_ROTATION_MAP, RegionType

        # Get window parameters
        window_params = all_parameters.get(RegionType.WINDOW.value, {})
        if not window_params:
            return all_parameters

        # Get window coordinates
        window_x1 = window_params.get(ParameterName.X1.value)
        window_y1 = window_params.get(ParameterName.Y1.value)
        window_x2 = window_params.get(ParameterName.X2.value)
        window_y2 = window_params.get(ParameterName.Y2.value)

        # Check window_geometry dict if individual coords not found
        if window_x1 is None and ParameterName.WINDOW_GEOMETRY.value in window_params:
            geom = window_params[ParameterName.WINDOW_GEOMETRY.value]
            window_x1 = geom.get(ParameterName.X1.value)
            window_y1 = geom.get(ParameterName.Y1.value)
            window_x2 = geom.get(ParameterName.X2.value)
            window_y2 = geom.get(ParameterName.Y2.value)

        if window_x1 is None:
            return all_parameters

        # Determine facade orientation
        window_geom = WindowGeometry(window_x1, window_y1, 0, window_x2, window_y2, 0)
        facade_angle = window_geom.get_facade_orientation()

        if abs(facade_angle) < 0.01:  # Already south-facing
            return all_parameters

        # Get rotation angle from map
        rotation_angle = FACADE_ROTATION_MAP.get(facade_angle, 0.0)
        origin = Point2D(0, 0)

        # Make a deep copy of parameters to avoid modifying original
        import copy
        rotated_params = copy.deepcopy(all_parameters)

        # Rotate window geometry
        rotated_window = window_geom.rotate(rotation_angle, origin)
        window_params_copy = rotated_params[RegionType.WINDOW.value]
        window_params_copy[ParameterName.X1.value] = rotated_window.x1
        window_params_copy[ParameterName.Y1.value] = rotated_window.y1
        window_params_copy[ParameterName.X2.value] = rotated_window.x2
        window_params_copy[ParameterName.Y2.value] = rotated_window.y2

        # Rotate room polygon if present
        room_params = rotated_params.get(RegionType.ROOM.value, {})
        if ParameterName.ROOM_POLYGON.value in room_params:
            polygon_data = room_params[ParameterName.ROOM_POLYGON.value]
            if isinstance(polygon_data, RoomPolygon):
                polygon = polygon_data
            else:
                polygon = RoomPolygon.from_dict(polygon_data)

            rotated_polygon = polygon.rotate(rotation_angle, origin)
            room_params[ParameterName.ROOM_POLYGON.value] = rotated_polygon

            # Also update window coordinates in room parameters
            room_params[ParameterName.X1.value] = rotated_window.x1
            room_params[ParameterName.Y1.value] = rotated_window.y1
            room_params[ParameterName.X2.value] = rotated_window.x2
            room_params[ParameterName.Y2.value] = rotated_window.y2

        return rotated_params

    def construct_from_flat_parameters(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Construct image from flat parameter dictionary

        Automatically groups parameters by region.
        Handles both legacy flat structure and unified windows structure.

        Args:
            model_type: The model type to use
            parameters: Dictionary of all parameters (flat or with windows)

        Returns:
            Complete encoded image
        """
        # Check if using unified structure with windows
        windows_key = ParameterName.WINDOWS.value
        if windows_key in parameters and isinstance(parameters[windows_key], dict):
            # Use multi-window construction which handles merging
            images_dict = self.construct_multi_window_images(model_type, parameters)
            # Return the first (and possibly only) image
            return next(iter(images_dict.values()))

        # Legacy flat structure - group and construct directly
        grouped = self._group_parameters(parameters)
        return self.construct_full_image(model_type, grouped)

    def construct_multi_window_images(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Construct multiple images for multiple windows in the same room

        Each window gets its own image with the room positioned relative to that window.
        Shared parameters (room, background) are reused across all images.

        Args:
            model_type: The model type to use
            parameters: Parameters including 'windows' dict with per-window configs

        Returns:
            Dictionary mapping window_id to encoded image
            Example: {"window_1": image1, "window_2": image2}
        """
        windows_key = ParameterName.WINDOWS.value
        if windows_key not in parameters:
            # Single window case - fallback to normal construction
            return {"window_1": self.construct_from_flat_parameters(model_type, parameters)}

        windows_config = parameters[windows_key]
        if not isinstance(windows_config, dict):
            raise ValueError("'windows' parameter must be a dictionary")

        # Extract shared parameters (everything except windows)
        shared_params = {k: v for k, v in parameters.items() if k != windows_key}

        # Build images using dict comprehension
        return {
            window_id: self.construct_from_flat_parameters(
                model_type,
                {**shared_params, **window_params}
            )
            for window_id, window_params in windows_config.items()
        }

    @staticmethod
    def _group_parameters(parameters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Group flat parameters by region using map (Strategy Pattern)

        Args:
            parameters: Flat parameter dictionary

        Returns:
            Parameters grouped by region
        """
        # Initialize grouped parameters
        grouped = {region.value: {} for region in RegionType}

        # Group parameters using map
        for key, value in parameters.items():
            region = PARAMETER_REGIONS.get(key)
            if region:
                grouped[region.value][key] = value

        # Room positioning depends on window coordinates - copy using list comprehension
        room_key = RegionType.ROOM.value
        window_key = RegionType.WINDOW.value
        background_key = RegionType.BACKGROUND.value

        # Copy window geometry to room
        coord_keys = [ParameterName.X1.value, ParameterName.Y1.value,
                      ParameterName.X2.value, ParameterName.Y2.value,
                      ParameterName.WINDOW_GEOMETRY.value]

        [grouped[room_key].update({k: grouped[window_key][k]})
         for k in coord_keys if k in grouped[window_key]]

        # Window sill height calculation depends on floor_height_above_terrain from background
        # Copy it to window region for calculator access
        if "floor_height_above_terrain" in grouped[background_key]:
            grouped[window_key]["floor_height_above_terrain"] = grouped[background_key]["floor_height_above_terrain"]

        return grouped
    