from typing import Dict, Any, Optional, Tuple
import copy
import math
import numpy as np
from src.components.interfaces import IImageBuilder, EncodingResult, EncodingParameters, RegionParameters
from src.components.enums import ModelType, RegionType, ParameterName, PARAMETER_REGIONS, EncodingScheme
from src.components.region_encoders import RegionEncoderFactory
from src.components.geometry import WindowGeometry, RoomPolygon, Point2D

class RoomImageBuilder(IImageBuilder):
    """
    Builder for creating encoded room images (Builder Pattern)

    Constructs 128×128 RGBA images with encoded room parameters
    """

    def __init__(self, encoding_scheme: EncodingScheme = EncodingScheme.RGB):
        self._image: Optional[np.ndarray] = None
        self._model_type: Optional[ModelType] = None
        self._encoding_scheme = encoding_scheme
        self._region_encoder_factory = RegionEncoderFactory()
        self._room_mask: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> 'RoomImageBuilder':
        """Reset builder to initial state"""
        # Create 128×128 RGBA image initialized to zeros
        self._image = np.zeros((128, 128, 4), dtype=np.uint8)
        self._model_type = None
        self._room_mask = None
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
        encoder = self._region_encoder_factory.get_encoder(region_type, self._encoding_scheme)
        self._image = encoder.encode_region(self._image, parameters, self._model_type)

        # Capture room mask when encoding room region
        if region_type == RegionType.ROOM:
            self._room_mask = encoder.get_last_mask()

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

    def get_room_mask(self) -> Optional[np.ndarray]:
        """
        Get the room mask (ones in room area, zeros elsewhere)

        Returns:
            128×128 single-channel mask or None if no room was encoded
        """
        if self._room_mask is None:
            return None
        return self._room_mask.copy()

    def _validate_state(self) -> None:
        """Validate builder state before operations"""
        if self._model_type is None:
            raise RuntimeError("Model type must be set before encoding")
        if self._image is None:
            raise RuntimeError("Image not initialized. Call reset() first.")


class ParameterNormalizer:
    """
    Normalizes parameters by converting dicts to proper geometry classes

    Ensures we always work with WindowGeometry and RoomPolygon classes, not dicts
    """

    @staticmethod
    def normalize_window_geometry(window_params: Dict[str, Any]) -> Optional[WindowGeometry]:
        """
        Extract or create WindowGeometry from parameters

        Args:
            window_params: Window parameter dictionary

        Returns:
            WindowGeometry object or None if no geometry found
        """
        # Check if window_geometry exists
        if ParameterName.WINDOW_GEOMETRY.value in window_params:
            geom = window_params[ParameterName.WINDOW_GEOMETRY.value]
            if isinstance(geom, WindowGeometry):
                return geom
            # Convert dict to WindowGeometry
            return WindowGeometry.from_dict(geom)

        # Check if individual coordinates exist
        if all(k in window_params for k in [
            ParameterName.X1.value, ParameterName.Y1.value, ParameterName.Z1.value,
            ParameterName.X2.value, ParameterName.Y2.value, ParameterName.Z2.value
        ]):
            return WindowGeometry(
                x1=window_params[ParameterName.X1.value],
                y1=window_params[ParameterName.Y1.value],
                z1=window_params[ParameterName.Z1.value],
                x2=window_params[ParameterName.X2.value],
                y2=window_params[ParameterName.Y2.value],
                z2=window_params[ParameterName.Z2.value],
                direction_angle=window_params.get(ParameterName.DIRECTION_ANGLE.value)
            )

        return None

    @staticmethod
    def normalize_room_polygon(room_params: Dict[str, Any]) -> Optional[RoomPolygon]:
        """
        Extract or create RoomPolygon from parameters

        Args:
            room_params: Room parameter dictionary

        Returns:
            RoomPolygon object or None if no polygon found
        """
        if ParameterName.ROOM_POLYGON.value not in room_params:
            return None

        polygon = room_params[ParameterName.ROOM_POLYGON.value]
        if isinstance(polygon, RoomPolygon):
            return polygon
        # RoomPolygon.from_dict() actually expects a list, not a dict (despite the name)
        if isinstance(polygon, (list, tuple)):
            return RoomPolygon.from_dict(polygon)
        # If it's a dict, it might be a serialized RoomPolygon
        if isinstance(polygon, dict) and 'vertices' in polygon:
            return RoomPolygon.from_dict(polygon['vertices'])
        return RoomPolygon.from_dict(polygon)


class GeometryRotator:
    """
    Rotates window and room geometry (Single Responsibility)
    """

    @staticmethod
    def rotate_if_needed(
        all_parameters: EncodingParameters,
        window_geom: WindowGeometry,
        room_polygon: Optional[RoomPolygon]
    ) -> EncodingParameters:
        """
        Rotate geometry if window is not pointing right (0 degrees)

        Args:
            all_parameters: All parameters grouped by region
            window_geom: WindowGeometry object with direction_angle set
            room_polygon: RoomPolygon object or None

        Returns:
            Parameters with rotated geometry
        """
        
        direction_angle_degrees = window_geom.direction_angle * 180 / math.pi  # Convert to degrees
        all_parameters.set_global(ParameterName.DIRECTION_ANGLE.value, window_geom.direction_angle)

        # If already pointing right (within tolerance), no rotation needed
        if abs(direction_angle_degrees) < 0.01:
            return all_parameters

        # Rotation angle is negative of direction angle (rotate opposite direction to align to 0°)
        rotation_angle = -direction_angle_degrees
        origin = Point2D(0, 0)

        # Calculate wall thickness BEFORE rotation (it's invariant under rotation)

        # Make a deep copy of parameters to avoid modifying original
        rotated_params = copy.deepcopy(all_parameters)

        # Rotate window geometry
        rotated_window = window_geom.rotate(rotation_angle, origin)
        window_params_copy = rotated_params.window.parameters

        GeometryRotator._update_window_coords(window_params_copy, rotated_window, window_geom.wall_thickness)

        # Rotate room polygon if present
        if room_polygon is not None:
            rotated_polygon = room_polygon.rotate(rotation_angle, origin)

            room_params = rotated_params.room.parameters
            room_params[ParameterName.ROOM_POLYGON.value] = rotated_polygon

            GeometryRotator._update_window_coords(room_params, rotated_window, window_geom.wall_thickness)

            # Set direction_angle to 0 after rotation (window now points right)
            room_params[ParameterName.DIRECTION_ANGLE.value] = 0.0

        return rotated_params

    @staticmethod
    def _update_window_coords(param_dict: dict, window: WindowGeometry, thickness: float) -> dict:
        """Update parameter dict with window coordinates and thickness"""
        param_dict[ParameterName.X1.value] = window.x1
        param_dict[ParameterName.Y1.value] = window.y1
        param_dict[ParameterName.X2.value] = window.x2
        param_dict[ParameterName.Y2.value] = window.y2
        param_dict[ParameterName.WALL_THICKNESS.value] = thickness
        return param_dict


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

        # Encode regions in order, snapping window to room facade edge
        for region in region_order:
            region_params = all_parameters.get_region(region).parameters
            if not region_params:
                continue

            # Before encoding window, snap its position to room's facade edge
            if region == RegionType.WINDOW:
                self._snap_window_to_room(all_parameters)

            self._builder.encode_region(region, region_params)

        # Build final image and get room mask
        return self._builder.build(), self._builder.get_room_mask()

    def _snap_window_to_room(self, all_parameters: EncodingParameters) -> None:
        """
        Find room polygon's facade right edge and inject it into window parameters.

        The window encoder uses this value to snap the window rectangle position
        so it starts exactly adjacent to the room polygon (no gap, no overlap),
        while preserving the window's pixel width.

        Only applies when a custom room polygon is present.
        """
        room_params = all_parameters.get_region(RegionType.ROOM).parameters
        if not room_params or ParameterName.ROOM_POLYGON.value not in room_params:
            return

        room_mask = self._builder.get_room_mask()
        if room_mask is None:
            return

        # Use center row (window center) to find facade edge
        center_y = room_mask.shape[0] // 2
        room_columns = np.where(room_mask[center_y])[0]
        if len(room_columns) == 0:
            return

        facade_right_edge = int(room_columns.max())

        window_params = all_parameters.get_region(RegionType.WINDOW).parameters
        if window_params:
            window_params['_room_facade_right_edge'] = facade_right_edge

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

        # NOTE: window_direction_angle is NOT auto-populated here because
        # the HSV override system relies on parameters being absent to apply
        # fixed defaults (e.g., DF models use constant 190 for orientation).
        # DA models must pass window_direction_angle explicitly in the caller
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
    