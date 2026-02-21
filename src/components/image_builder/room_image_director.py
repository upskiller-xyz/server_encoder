from typing import Dict, Any, Optional, Tuple
from src.core import ModelType, RegionType, ParameterName, PARAMETER_REGIONS
from src.models import EncodingParameters, EncodingResult
from src.components.image_builder.room_image_builder import RoomImageBuilder
from src.components.image_builder.parameter_normalizer import ParameterNormalizer
from src.components.image_builder.geometry_rotator import GeometryRotator
from src.components.geometry import WindowGeometry
import numpy as np


class RoomImageDirector:
    """
    Director class for orchestrating image building (Director Pattern)

    Provides high-level interface for building complete encoded images
    """

    def __init__(self, builder: RoomImageBuilder):
        self._builder = builder

    def construct_full_image(
        self,
        model_type: ModelType,
        all_parameters: EncodingParameters
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Construct a complete encoded image with all regions

        Args:
            model_type: The model type to use
            all_parameters: All parameters grouped by region

        Returns:
            Tuple of (encoded_image, room_mask)
            - encoded_image: Complete encoded image
            - room_mask: Room mask (ones in room area, zeros elsewhere) or None
        """
        # Reset and configure builder
        self._builder.reset().set_model_type(model_type)

        # Rotate geometry if window is not on south facade
        all_parameters = self._rotate_geometry_if_needed(all_parameters)

        # Define region encoding order
        region_order = [
            RegionType.BACKGROUND,
            RegionType.ROOM,
            RegionType.WINDOW,
            RegionType.OBSTRUCTION_BAR
        ]
        params = [all_parameters.get_region(region).parameters for region in region_order]
        # Encode regions in order using list comprehension
        # EncodingParameters.from_dict(p) instead of p
        [self._builder.encode_region(region, p)
         for region, p in zip(region_order, params)
         if p]

        # Build final image and get room mask
        return self._builder.build(), self._builder.get_room_mask()

    def _rotate_geometry_if_needed(self, all_parameters: EncodingParameters) -> EncodingParameters:
        """
        Rotate room polygon and window coordinates if window is not on south facade

        Uses helper classes to maintain Single Responsibility Principle

        Args:
            all_parameters: Parameters grouped by region

        Returns:
            Parameters with rotated geometry
        """
        # Get window parameters and normalize to WindowGeometry
        if not all_parameters.window.parameters:
            return all_parameters

        window_geom = ParameterNormalizer.normalize_window_geometry(all_parameters.window.parameters)
        if window_geom is None:
            return all_parameters

        # Get room polygon
        room_polygon = ParameterNormalizer.normalize_room_polygon(all_parameters.room.parameters) if all_parameters.room.parameters else None

        # Calculate direction_angle from room polygon if available and not already set
        if window_geom.direction_angle is None and room_polygon is not None:
            try:
                calculated_angle = window_geom.calculate_direction_from_polygon(room_polygon)
                # Create new WindowGeometry with updated direction_angle
                window_geom = WindowGeometry(
                    x1=window_geom.x1, y1=window_geom.y1, z1=window_geom.z1,
                    x2=window_geom.x2, y2=window_geom.y2, z2=window_geom.z2,
                    direction_angle=calculated_angle
                )
            except ValueError:
                pass

        direction_angle = window_geom.direction_angle

        # Propagate direction_angle to all relevant regions
        all_parameters.set_global(ParameterName.DIRECTION_ANGLE.value, direction_angle)
        all_parameters.window[ParameterName.DIRECTION_ANGLE.value] = direction_angle
        if all_parameters.room.parameters:
            all_parameters.room[ParameterName.DIRECTION_ANGLE.value] = direction_angle

        # NOTE: window_orientation is NOT auto-populated here because
        # the HSV override system relies on parameters being absent to apply
        # fixed defaults (e.g., DF models use constant 190 for orientation).
        # DA models must pass window_orientation explicitly in the caller
        # (training pipeline or server_lux encoding handler).

        # Rotate geometry using GeometryRotator
        return GeometryRotator.rotate_if_needed(all_parameters, window_geom, room_polygon)

    def construct_from_flat_parameters(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Construct image from flat parameter dictionary

        Automatically groups parameters by region.
        Handles both legacy flat structure and unified windows structure.

        Args:
            model_type: The model type to use
            parameters: Dictionary of all parameters (flat or with windows)

        Returns:
            Tuple of (encoded_image, room_mask)
        """
        # Check if using unified structure with windows
        windows_key = ParameterName.WINDOWS.value
        if windows_key in parameters and isinstance(parameters[windows_key], dict):
            # Use multi-window construction which handles merging
            result = self.construct_multi_window_images(model_type, parameters)
            # Return the first (and possibly only) image and mask
            return result.get_first_image(), result.get_first_mask()

        # Legacy flat structure - group and construct directly
        grouped = self._group_parameters(parameters)

        return self.construct_full_image(model_type, grouped)

    def construct_multi_window_images(
        self,
        model_type: ModelType,
        parameters: Dict[str, Any]
    ) -> EncodingResult:
        """
        Construct multiple images for multiple windows in the same room

        Each window gets its own image with the room positioned relative to that window.
        Shared parameters (room, background) are reused across all images.

        Args:
            model_type: The model type to use
            parameters: Parameters including 'windows' dict with per-window configs

        Returns:
            EncodingResult containing images and masks for all windows
        """
        result = EncodingResult()

        windows_key = ParameterName.WINDOWS.value
        if windows_key not in parameters:
            # Single window case - fallback to normal construction
            image, mask = self.construct_from_flat_parameters(model_type, parameters)
            result.add_window("window_1", image, mask)
            return result

        windows_config = parameters[windows_key]
        if not isinstance(windows_config, dict):
            raise ValueError("'windows' parameter must be a dictionary")

        # Extract shared parameters (everything except windows)
        shared_params = {k: v for k, v in parameters.items() if k != windows_key}

        # Build images and masks
        for window_id, window_params in windows_config.items():
            image, mask = self.construct_from_flat_parameters(
                model_type,
                {**shared_params, **window_params}
            )
            result.add_window(window_id, image, mask)

        return result

    @staticmethod
    def _group_parameters(parameters: Dict[str, Any]) -> EncodingParameters:
        """
        Group flat parameters by region and normalize all geometry to proper classes

        Args:
            parameters: Flat parameter dictionary

        Returns:
            EncodingParameters with normalized geometry
        """
        # Initialize encoding parameters
        encoding_params = EncodingParameters()

        # Group parameters by region
        for key, value in parameters.items():
            region = PARAMETER_REGIONS.get(key)
            if region:
                encoding_params.get_region(region)[key] = value

        # NORMALIZE GEOMETRY: Convert dicts to proper classes at entry point
        # This ensures we ALWAYS work with WindowGeometry and RoomPolygon classes internally

        # Normalize WindowGeometry
        window_params = encoding_params.window.parameters
        if window_params:
            window_geom = ParameterNormalizer.normalize_window_geometry(window_params)
            if window_geom:
                # Replace dict data with WindowGeometry object
                window_params[ParameterName.WINDOW_GEOMETRY.value] = window_geom
                # Also set individual coords for backwards compatibility
                window_params[ParameterName.X1.value] = window_geom.x1
                window_params[ParameterName.Y1.value] = window_geom.y1
                window_params[ParameterName.Z1.value] = window_geom.z1
                window_params[ParameterName.X2.value] = window_geom.x2
                window_params[ParameterName.Y2.value] = window_geom.y2
                window_params[ParameterName.Z2.value] = window_geom.z2
                if window_geom.direction_angle is not None:
                    window_params[ParameterName.DIRECTION_ANGLE.value] = window_geom.direction_angle

        # Normalize RoomPolygon
        room_params = encoding_params.room.parameters
        if room_params:
            room_polygon = ParameterNormalizer.normalize_room_polygon(room_params)
            if room_polygon:
                # Replace dict data with RoomPolygon object
                room_params[ParameterName.ROOM_POLYGON.value] = room_polygon

        # Copy window geometry to room
        coord_keys = [ParameterName.X1.value, ParameterName.Y1.value,
                      ParameterName.X2.value, ParameterName.Y2.value,
                      ParameterName.WINDOW_GEOMETRY.value, ParameterName.DIRECTION_ANGLE.value,
                      ParameterName.WALL_THICKNESS.value]

        for k in coord_keys:
            if k in window_params:
                room_params[k] = window_params[k]

        # Window sill height calculation depends on floor_height_above_terrain from background
        # Copy it to window region for calculator access
        background_params = encoding_params.background.parameters
        if ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value in background_params:
            window_params[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value] = background_params[ParameterName.FLOOR_HEIGHT_ABOVE_TERRAIN.value]

        return encoding_params
